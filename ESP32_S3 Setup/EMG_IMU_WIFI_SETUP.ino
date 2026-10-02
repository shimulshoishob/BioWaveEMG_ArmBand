/*
 * ESP32-S3 8CH EMG + BNO080 IMU Wi-Fi Streamer
 *
 * What this firmware does:
 * - USB serial provisioning for router Wi-Fi credentials
 * - 8-channel raw EMG sampling at 500 Hz
 * - BNO080 fused Pitch/Roll/Yaw updates with a 5 ms request
 *   for the 200 Hz target mode used by this project
 * - Latest IMU orientation is attached to every 500 Hz EMG frame
 * - Wi-Fi is used only for discovery, control, and UDP streaming
 * - Dual-core task split:
 *   Core 1: time-critical 500 Hz EMG acquisition
 *   Core 0: IMU polling, USB serial commands, Wi-Fi control, UDP streaming
 *
 * Important:
 * - This streams a NEW binary packet format, not the 16-byte IMU-only packet
 *   used by the current desktop app. The desktop app will need a matching
 *   parser for this combined EMG+IMU format.
 * - EMG sensor signal outputs must stay within the ESP32-S3 ADC safe range
 *   (0..3.3 V on the ADC pins), even if the EMG modules are powered from 5 V.
 *
 * UDP transport uses batching to avoid sending hundreds of tiny packets
 * per second. Each UDP packet starts with a small header followed by 5
 * consecutive frames.
 *
 * Header:
 *   char     magic[4]   = "BWIM"
 *   uint8_t  version    = 1
 *   uint8_t  frameCount = 5
 *   uint16_t frameSize  = 48
 *   uint32_t packetSequence
 *
 * Frames:
 *   5 x {
 *     uint32_t frameId
 *     uint32_t frameTimestampUs
 *     uint32_t imuSampleId
 *     uint32_t imuTimestampUs
 *     uint16_t emg[8]
 *     float    roll
 *     float    pitch
 *     float    yaw
 *     uint8_t  imuFresh
 *     uint8_t  reserved[3]
 *   }
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <SPI.h>
#include <Preferences.h>
#include <esp_system.h>
#include <mbedtls/md.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>
#include "SparkFun_BNO08x_Arduino_Library.h"

// ---------- NETWORK / CONTROL ----------
static const uint16_t STREAM_PORT = 5000;
static const uint16_t CONTROL_PORT = 5001;
static const unsigned long WIFI_CONNECT_TIMEOUT_MS = 15000;
static const unsigned long WIFI_RETRY_INTERVAL_MS = 10000;
static const unsigned long STREAM_KEEPALIVE_TIMEOUT_MS = 5000;
static const unsigned long CHALLENGE_TTL_MS = 30000;
static const char *FW_VERSION = "emg-imu-1.6.0";

// Change this before flashing.
static const char *DEVICE_ACCESS_KEY = "CHANGE_THIS_TO_A_LONG_RANDOM_KEY";

// ---------- EMG ----------
static const uint8_t EMG_CHANNELS = 8;
static const int EMG_PINS[EMG_CHANNELS] = {1, 2, 4, 5, 6, 7, 9, 10};
static const uint32_t EMG_FRAME_RATE_HZ = 500;
static const uint32_t EMG_FRAME_PERIOD_US = 1000000UL / EMG_FRAME_RATE_HZ; // 2000 us
static const uint8_t EMG_FRAMES_PER_PACKET = 5;

// ---------- IMU ----------
// The SparkFun helper uses millisecond scheduling. A 5 ms request is the
// practical 200 Hz target mode for this project on the BNO080.
static const uint16_t IMU_REPORT_INTERVAL_MS = 5; // 200 Hz requested target

// ---------- QUEUE / TASKING ----------
static const uint16_t FRAME_QUEUE_LENGTH = 64;
// On ESP32-S3/Arduino, Wi-Fi and system tasks live primarily on Core 0.
// Keep the time-critical acquisition loop on Core 1 so it is disturbed less.
static const BaseType_t ACQUISITION_CORE = 1;
static const BaseType_t NETWORK_CORE = 0;
static const UBaseType_t ACQUISITION_PRIORITY = 3;
static const UBaseType_t NETWORK_PRIORITY = 2;
static const UBaseType_t IMU_PRIORITY = 2;
static const uint8_t MAX_UDP_PACKETS_PER_PASS = 8;
static const uint16_t SHORT_BUSY_WAIT_US = 75;
static const char *CREDENTIAL_PREFIX = "enc1";

// ---------- IMU PINS ----------
#define BNO_RST  11
#define BNO_INT  12
#define BNO_CS   13
#define BNO_MOSI 14
#define BNO_MISO 15
#define BNO_SCK  16

// ---------- FORWARD DECLARATIONS ----------
void handleSerialCommands();
bool connectToSavedWiFi();
void monitorWiFiState();
void handleControlPacket();
void printDiagnosticsIfDue();
void flushFrameQueue();
void sendFrameUdp(const void *frameData, size_t frameSize);
void updateLatestImu();
String randomHex(uint32_t value);
String hmacSha256Hex(const String &message, const String &key);
String getToken(const String &message, int index);

WiFiUDP udp;
Preferences preferences;
BNO08x myIMU;
QueueHandle_t frameQueue = nullptr;
TaskHandle_t acquisitionTaskHandle = nullptr;
TaskHandle_t networkTaskHandle = nullptr;
TaskHandle_t imuTaskHandle = nullptr;
portMUX_TYPE imuMutex = portMUX_INITIALIZER_UNLOCKED;

volatile bool targetLocked = false;
volatile unsigned long lastAuthorizedCommandMs = 0;
volatile unsigned long lastWiFiRetryMs = 0;
IPAddress streamTargetIP;
uint16_t streamTargetPort = STREAM_PORT;

String deviceId;
String deviceName;
String pendingChallenge;
unsigned long challengeIssuedAt = 0;
bool imuReady = false;

struct LatestImuState {
  float roll;
  float pitch;
  float yaw;
  uint32_t sampleId;
  uint32_t timestampUs;
};

struct __attribute__((packed)) EmgImuFrame {
  uint32_t frameId;
  uint32_t frameTimestampUs;
  uint32_t imuSampleId;
  uint32_t imuTimestampUs;
  uint16_t emg[EMG_CHANNELS];
  float roll;
  float pitch;
  float yaw;
  uint8_t imuFresh;
  uint8_t reserved[3];
};

struct __attribute__((packed)) PacketHeader {
  char magic[4];
  uint8_t version;
  uint8_t frameCount;
  uint16_t frameSize;
  uint32_t packetSequence;
};

struct __attribute__((packed)) EmgImuPacket {
  PacketHeader header;
  EmgImuFrame frames[EMG_FRAMES_PER_PACKET];
};

LatestImuState latestImu = {0.0f, 0.0f, 0.0f, 0, 0};
uint32_t packetSequenceCounter = 0;

volatile uint32_t emgFramesThisWindow = 0;
volatile uint32_t imuEventsThisWindow = 0;
volatile uint32_t udpPacketsSentThisWindow = 0;
volatile uint32_t queueDropsThisWindow = 0;
volatile uint32_t acquisitionOverrunsThisWindow = 0;
unsigned long diagnosticsWindowStartMs = 0;

static inline int readADCStable(int pin) {
  // Throw away the first reading after the ADC mux switch to reduce artifacts.
  (void)analogRead(pin);
  return analogRead(pin);
}

uint8_t hexNibble(char c) {
  if (c >= '0' && c <= '9') {
    return static_cast<uint8_t>(c - '0');
  }
  if (c >= 'a' && c <= 'f') {
    return static_cast<uint8_t>(10 + (c - 'a'));
  }
  if (c >= 'A' && c <= 'F') {
    return static_cast<uint8_t>(10 + (c - 'A'));
  }
  return 0;
}

String protectStoredValue(const String &label, const String &plainText) {
  if (plainText.length() == 0) {
    return "";
  }

  String nonce = randomHex(esp_random()) + randomHex(esp_random());
  String cipherHex;
  cipherHex.reserve(plainText.length() * 2);

  for (size_t offset = 0; offset < plainText.length(); offset += 32) {
    String maskHex = hmacSha256Hex(label + "|" + deviceId + "|" + nonce + "|" + String(offset / 32), DEVICE_ACCESS_KEY);
    size_t blockLength = min(static_cast<size_t>(32), plainText.length() - offset);
    for (size_t i = 0; i < blockLength; ++i) {
      uint8_t maskByte = static_cast<uint8_t>((hexNibble(maskHex[i * 2]) << 4) | hexNibble(maskHex[i * 2 + 1]));
      uint8_t cipherByte = static_cast<uint8_t>(plainText[offset + i]) ^ maskByte;
      char pair[3];
      snprintf(pair, sizeof(pair), "%02x", cipherByte);
      cipherHex += pair;
    }
  }

  return String(CREDENTIAL_PREFIX) + "|" + nonce + "|" + cipherHex;
}

String unprotectStoredValue(const String &label, const String &storedValue) {
  if (!storedValue.startsWith(String(CREDENTIAL_PREFIX) + "|")) {
    return storedValue;
  }

  String nonce = getToken(storedValue, 1);
  String cipherHex = getToken(storedValue, 2);
  if (nonce.length() == 0 || cipherHex.length() == 0 || (cipherHex.length() % 2) != 0) {
    return "";
  }

  String plainText;
  plainText.reserve(cipherHex.length() / 2);
  size_t cipherBytes = cipherHex.length() / 2;

  for (size_t offset = 0; offset < cipherBytes; offset += 32) {
    String maskHex = hmacSha256Hex(label + "|" + deviceId + "|" + nonce + "|" + String(offset / 32), DEVICE_ACCESS_KEY);
    size_t blockLength = min(static_cast<size_t>(32), cipherBytes - offset);
    for (size_t i = 0; i < blockLength; ++i) {
      size_t hexIndex = (offset + i) * 2;
      uint8_t cipherByte = static_cast<uint8_t>((hexNibble(cipherHex[hexIndex]) << 4) | hexNibble(cipherHex[hexIndex + 1]));
      uint8_t maskByte = static_cast<uint8_t>((hexNibble(maskHex[i * 2]) << 4) | hexNibble(maskHex[i * 2 + 1]));
      plainText += static_cast<char>(cipherByte ^ maskByte);
    }
  }

  return plainText;
}

String buildDeviceId() {
  uint64_t chipId = ESP.getEfuseMac();
  char buffer[13];
  snprintf(buffer, sizeof(buffer), "%04X%08X", (uint16_t)(chipId >> 32), (uint32_t)chipId);
  return String(buffer);
}

void restartControlSocket() {
  udp.stop();
  udp.begin(CONTROL_PORT);
}

void clearStreamTarget() {
  targetLocked = false;
  streamTargetIP = IPAddress();
  streamTargetPort = STREAM_PORT;
  flushFrameQueue();
}

void saveWiFiCredentials(const String &ssid, const String &password) {
  preferences.begin("wifi", false);
  preferences.putString("ssid", protectStoredValue("ssid", ssid));
  preferences.putString("pass", protectStoredValue("pass", password));
  preferences.end();
}

void loadWiFiCredentials(String &ssid, String &password) {
  preferences.begin("wifi", true);
  ssid = unprotectStoredValue("ssid", preferences.getString("ssid", ""));
  password = unprotectStoredValue("pass", preferences.getString("pass", ""));
  preferences.end();
}

bool hasSavedWiFiCredentials() {
  String ssid;
  String password;
  loadWiFiCredentials(ssid, password);
  return ssid.length() > 0;
}

String currentWiFiModeString() {
  return "STA";
}

String currentReachableIpString() {
  return WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString() : String("0.0.0.0");
}

void sendReply(const IPAddress &remoteIp, uint16_t remotePort, const String &message) {
  udp.beginPacket(remoteIp, remotePort);
  udp.print(message);
  udp.endPacket();
}

String randomHex(uint32_t value) {
  char buffer[9];
  snprintf(buffer, sizeof(buffer), "%08lx", static_cast<unsigned long>(value));
  return String(buffer);
}

String generateChallenge() {
  return randomHex(esp_random()) + randomHex(esp_random());
}

String hmacSha256Hex(const String &message, const String &key) {
  const mbedtls_md_info_t *mdInfo = mbedtls_md_info_from_type(MBEDTLS_MD_SHA256);
  unsigned char hmacOutput[32];
  char hexOutput[65];

  mbedtls_md_hmac(
    mdInfo,
    reinterpret_cast<const unsigned char *>(key.c_str()),
    key.length(),
    reinterpret_cast<const unsigned char *>(message.c_str()),
    message.length(),
    hmacOutput
  );

  for (int i = 0; i < 32; ++i) {
    snprintf(&hexOutput[i * 2], 3, "%02x", hmacOutput[i]);
  }
  hexOutput[64] = '\0';
  return String(hexOutput);
}

bool secureEquals(const String &a, const String &b) {
  if (a.length() != b.length()) {
    return false;
  }

  uint8_t diff = 0;
  for (size_t i = 0; i < a.length(); ++i) {
    diff |= static_cast<uint8_t>(a[i] ^ b[i]);
  }
  return diff == 0;
}

String urlDecode(const String &input) {
  String output;
  output.reserve(input.length());

  for (size_t i = 0; i < input.length(); ++i) {
    char c = input[i];
    if (c == '%' && (i + 2) < input.length()) {
      char hi = input[i + 1];
      char lo = input[i + 2];
      char hex[3] = {hi, lo, '\0'};
      output += static_cast<char>(strtol(hex, nullptr, 16));
      i += 2;
    } else if (c == '+') {
      output += ' ';
    } else {
      output += c;
    }
  }
  return output;
}

String getToken(const String &message, int index) {
  int start = 0;
  int current = 0;

  while (current < index) {
    start = message.indexOf('|', start);
    if (start == -1) {
      return "";
    }
    start += 1;
    current++;
  }

  int end = message.indexOf('|', start);
  if (end == -1) {
    end = message.length();
  }
  return message.substring(start, end);
}

int countTokens(const String &message) {
  if (message.length() == 0) {
    return 0;
  }

  int count = 1;
  for (size_t i = 0; i < message.length(); ++i) {
    if (message[i] == '|') {
      count++;
    }
  }
  return count;
}

String helloPacket() {
  return "HELLO|" + deviceId + "|" + deviceName + "|" + currentWiFiModeString() + "|" +
         currentReachableIpString() + "|" + String(imuReady ? 1 : 0) + "|" +
         String(targetLocked ? 1 : 0) + "|" + FW_VERSION;
}

bool enableOrientationReport() {
  return myIMU.enableRotationVector(IMU_REPORT_INTERVAL_MS);
}

bool connectToSavedWiFi() {
  String savedSsid;
  String savedPass;
  loadWiFiCredentials(savedSsid, savedPass);

  if (savedSsid.length() == 0) {
    udp.stop();
    return false;
  }

  Serial.println();
  Serial.println("====================================");
  Serial.print("Connecting to saved Wi-Fi: ");
  Serial.println(savedSsid);
  Serial.println("====================================");

  clearStreamTarget();
  WiFi.disconnect(true, true);
  delay(200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(savedSsid.c_str(), savedPass.c_str());

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - startedAt) < WIFI_CONNECT_TIMEOUT_MS) {
    handleSerialCommands();
    delay(50);
    handleSerialCommands();
    delay(50);
    handleSerialCommands();
    delay(50);
    handleSerialCommands();
    delay(50);
    handleSerialCommands();
    delay(50);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() != WL_CONNECTED) {
    udp.stop();
    Serial.println("Wi-Fi connection failed. Waiting for USB provisioning or later retry.");
    return false;
  }

  restartControlSocket();
  Serial.print("Connected. IP address: ");
  Serial.println(WiFi.localIP());
  return true;
}

void initImu() {
  SPI.begin(BNO_SCK, BNO_MISO, BNO_MOSI, BNO_CS);

  Serial.println("Performing BNO08x hard reset...");
  pinMode(BNO_RST, OUTPUT);
  digitalWrite(BNO_RST, LOW);
  delay(10);
  digitalWrite(BNO_RST, HIGH);
  delay(100);

  Serial.println("Starting BNO08x over SPI...");
  if (!myIMU.beginSPI(BNO_CS, BNO_INT, BNO_RST, 1000000, SPI)) {
    Serial.println("ERROR: BNO08x not detected.");
    imuReady = false;
    return;
  }

  delay(500);

  for (int attempt = 0; attempt < 5; ++attempt) {
    if (enableOrientationReport()) {
      imuReady = true;
      latestImu.sampleId = 0;
      latestImu.timestampUs = micros();
      diagnosticsWindowStartMs = millis();
      Serial.println("Rotation Vector enabled with a 200 Hz request.");
      return;
    }
    Serial.println("Retrying Rotation Vector enable...");
    delay(300);
  }

  Serial.println("ERROR: Failed to enable Rotation Vector.");
  imuReady = false;
}

bool verifyAuthorizedCommand(const String &command, const String &challenge, const String &payload, const String &authHex) {
  if (pendingChallenge.length() == 0) {
    return false;
  }
  if ((millis() - challengeIssuedAt) > CHALLENGE_TTL_MS) {
    pendingChallenge = "";
    return false;
  }
  if (challenge != pendingChallenge) {
    return false;
  }

  String message = command + "|" + challenge;
  if (payload.length() > 0) {
    message += "|" + payload;
  }

  String expectedAuth = hmacSha256Hex(message, DEVICE_ACCESS_KEY);
  bool ok = secureEquals(expectedAuth, authHex);
  pendingChallenge = "";

  if (ok) {
    lastAuthorizedCommandMs = millis();
  }
  return ok;
}

void handleStart(const String &message, const IPAddress &remoteIp, uint16_t remotePort) {
  if (countTokens(message) != 5) {
    sendReply(remoteIp, remotePort, "ERR|BAD_FORMAT");
    return;
  }

  String challenge = getToken(message, 1);
  String clientIpText = getToken(message, 2);
  String clientPortText = getToken(message, 3);
  String authHex = getToken(message, 4);

  if (!verifyAuthorizedCommand("START", challenge, clientIpText + "|" + clientPortText, authHex)) {
    sendReply(remoteIp, remotePort, "ERR|AUTH_FAILED");
    return;
  }

  IPAddress requestedTargetIp;
  if (!requestedTargetIp.fromString(clientIpText)) {
    sendReply(remoteIp, remotePort, "ERR|BAD_IP");
    return;
  }

  int requestedPort = clientPortText.toInt();
  if (requestedPort <= 0 || requestedPort > 65535) {
    sendReply(remoteIp, remotePort, "ERR|BAD_PORT");
    return;
  }

  streamTargetIP = requestedTargetIp;
  streamTargetPort = static_cast<uint16_t>(requestedPort);
  targetLocked = true;

  Serial.println("====================================");
  Serial.println("Streaming target locked");
  Serial.print("Client IP: ");
  Serial.println(streamTargetIP);
  Serial.print("Client Port: ");
  Serial.println(streamTargetPort);
  Serial.println("====================================");

  flushFrameQueue();
  sendReply(remoteIp, remotePort, "ACK|STARTED");
}

void handleStop(const String &message, const IPAddress &remoteIp, uint16_t remotePort) {
  if (countTokens(message) != 3) {
    sendReply(remoteIp, remotePort, "ERR|BAD_FORMAT");
    return;
  }

  String challenge = getToken(message, 1);
  String authHex = getToken(message, 2);

  if (!verifyAuthorizedCommand("STOP", challenge, "", authHex)) {
    sendReply(remoteIp, remotePort, "ERR|AUTH_FAILED");
    return;
  }

  clearStreamTarget();
  sendReply(remoteIp, remotePort, "ACK|STOPPED");
}

void handlePing(const String &message, const IPAddress &remoteIp, uint16_t remotePort) {
  if (countTokens(message) != 3) {
    sendReply(remoteIp, remotePort, "ERR|BAD_FORMAT");
    return;
  }

  String challenge = getToken(message, 1);
  String authHex = getToken(message, 2);

  if (!verifyAuthorizedCommand("PING", challenge, "", authHex)) {
    sendReply(remoteIp, remotePort, "ERR|AUTH_FAILED");
    return;
  }

  sendReply(remoteIp, remotePort, "ACK|PONG");
}

void handleControlPacket() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  int packetSize = udp.parsePacket();
  if (!packetSize) {
    return;
  }

  char incomingBuffer[384];
  int len = udp.read(incomingBuffer, sizeof(incomingBuffer) - 1);
  if (len <= 0) {
    return;
  }
  incomingBuffer[len] = '\0';

  String message = String(incomingBuffer);
  IPAddress remoteIp = udp.remoteIP();
  uint16_t remotePort = udp.remotePort();

  if (message == "DISCOVER") {
    sendReply(remoteIp, remotePort, helloPacket());
    return;
  }

  if (message == "CHALLENGE") {
    pendingChallenge = generateChallenge();
    challengeIssuedAt = millis();
    sendReply(remoteIp, remotePort, "CHALLENGE|" + pendingChallenge);
    return;
  }

  String command = getToken(message, 0);
  if (command == "START") {
    handleStart(message, remoteIp, remotePort);
  } else if (command == "STOP") {
    handleStop(message, remoteIp, remotePort);
  } else if (command == "PING") {
    handlePing(message, remoteIp, remotePort);
  } else {
    sendReply(remoteIp, remotePort, "ERR|UNKNOWN_COMMAND");
  }
}

void processSerialCommand(const String &line) {
  if (line.length() == 0) {
    return;
  }

  if (line == "INFO") {
    Serial.println(
      "INFO|" + deviceId + "|" + deviceName + "|" + String(imuReady ? 1 : 0) + "|" +
      String(hasSavedWiFiCredentials() ? 1 : 0) + "|" + FW_VERSION
    );
    return;
  }

  if (line.startsWith("PROVISION|")) {
    if (countTokens(line) != 3) {
      Serial.println("ERR|BAD_FORMAT");
      return;
    }

    String encodedSsid = getToken(line, 1);
    String encodedPassword = getToken(line, 2);
    String ssid = urlDecode(encodedSsid);
    String password = urlDecode(encodedPassword);

    if (ssid.length() == 0) {
      Serial.println("ERR|MISSING_SSID");
      return;
    }

    saveWiFiCredentials(ssid, password);
    Serial.println("ACK|PROVISIONED");
    delay(1500);
    ESP.restart();
    return;
  }

  Serial.println("ERR|UNKNOWN_COMMAND");
}

void handleSerialCommands() {
  static String inputBuffer;

  while (Serial.available() > 0) {
    char incoming = static_cast<char>(Serial.read());
    if (incoming == '\n' || incoming == '\r') {
      if (inputBuffer.length() > 0) {
        processSerialCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += incoming;
      if (inputBuffer.length() > 255) {
        inputBuffer = "";
      }
    }
  }
}

void updateLatestImu() {
  if (!imuReady) {
    return;
  }

  int maxDrain = 12;
  bool sawNewImu = false;
  LatestImuState tempImu = latestImu;

  while (myIMU.getSensorEvent() && maxDrain > 0) {
    maxDrain--;
    if (myIMU.getSensorEventID() != SENSOR_REPORTID_ROTATION_VECTOR) {
      continue;
    }

    tempImu.roll = myIMU.getRoll() * (180.0f / PI);
    tempImu.pitch = myIMU.getPitch() * (180.0f / PI);
    tempImu.yaw = myIMU.getYaw() * (180.0f / PI);
    tempImu.timestampUs = micros();
    imuEventsThisWindow++;
    sawNewImu = true;
  }

  if (!sawNewImu) {
    return;
  }

  portENTER_CRITICAL(&imuMutex);
  latestImu.roll = tempImu.roll;
  latestImu.pitch = tempImu.pitch;
  latestImu.yaw = tempImu.yaw;
  latestImu.timestampUs = tempImu.timestampUs;
  latestImu.sampleId++;
  portEXIT_CRITICAL(&imuMutex);
}

void flushFrameQueue() {
  if (frameQueue == nullptr) {
    return;
  }

  EmgImuPacket throwaway;
  while (xQueueReceive(frameQueue, &throwaway, 0) == pdTRUE) {
    // Drain stale frames.
  }
}

void queueFrame(const EmgImuPacket &packet) {
  if (frameQueue == nullptr) {
    return;
  }

  if (xQueueSendToBack(frameQueue, &packet, 0) == pdTRUE) {
    return;
  }

  EmgImuPacket throwaway;
  (void)xQueueReceive(frameQueue, &throwaway, 0);
  if (xQueueSendToBack(frameQueue, &packet, 0) != pdTRUE) {
    // If this still fails, the consumer is badly stalled; keep going.
  }
  queueDropsThisWindow++;
}

void sendFrameUdp(const void *frameData, size_t frameSize) {
  udp.beginPacket(streamTargetIP, streamTargetPort);
  udp.write(reinterpret_cast<const uint8_t *>(frameData), frameSize);
  udp.endPacket();
  udpPacketsSentThisWindow++;
}

void printDiagnosticsIfDue() {
  unsigned long nowMs = millis();
  if (diagnosticsWindowStartMs == 0) {
    diagnosticsWindowStartMs = nowMs;
    return;
  }

  if ((nowMs - diagnosticsWindowStartMs) < 1000) {
    return;
  }

  float seconds = (nowMs - diagnosticsWindowStartMs) / 1000.0f;
  float emgRateHz = emgFramesThisWindow / seconds;
  float imuRateHz = imuEventsThisWindow / seconds;
  float udpRateHz = udpPacketsSentThisWindow / seconds;

  Serial.print("EMG Frame Rate: ");
  Serial.print(emgRateHz, 1);
  Serial.print(" Hz | IMU Event Rate: ");
  Serial.print(imuRateHz, 1);
  Serial.print(" Hz | UDP Send Rate: ");
  Serial.print(udpRateHz, 1);
  Serial.print(" Hz | Queue Drops: ");
  Serial.print(queueDropsThisWindow);
  Serial.print(" | Acquisition Overruns: ");
  Serial.println(acquisitionOverrunsThisWindow);

  emgFramesThisWindow = 0;
  imuEventsThisWindow = 0;
  udpPacketsSentThisWindow = 0;
  queueDropsThisWindow = 0;
  acquisitionOverrunsThisWindow = 0;
  diagnosticsWindowStartMs = nowMs;
}

void monitorWiFiState() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  if (!hasSavedWiFiCredentials()) {
    return;
  }

  if ((millis() - lastWiFiRetryMs) < WIFI_RETRY_INTERVAL_MS) {
    return;
  }

  lastWiFiRetryMs = millis();
  Serial.println("Retrying saved Wi-Fi connection...");
  connectToSavedWiFi();
}

void acquisitionTask(void *parameter) {
  EmgImuFrame frame;
  EmgImuPacket packet;
  memset(&frame, 0, sizeof(frame));
  memset(&packet, 0, sizeof(packet));
  memcpy(packet.header.magic, "BWIM", 4);
  packet.header.version = 1;
  packet.header.frameCount = EMG_FRAMES_PER_PACKET;
  packet.header.frameSize = sizeof(EmgImuFrame);
  packet.header.packetSequence = 0;

  uint32_t frameId = 0;
  uint32_t nextSampleUs = micros();
  uint32_t lastPackedSampleId = 0;
  uint8_t packetFrameIndex = 0;

  for (;;) {
    LatestImuState currentImu;
    portENTER_CRITICAL(&imuMutex);
    currentImu = latestImu;
    portEXIT_CRITICAL(&imuMutex);

    frame.frameId = frameId++;
    frame.frameTimestampUs = micros();

    for (int ch = 0; ch < EMG_CHANNELS; ++ch) {
      frame.emg[ch] = static_cast<uint16_t>(readADCStable(EMG_PINS[ch]));
    }

    frame.imuSampleId = currentImu.sampleId;
    frame.imuTimestampUs = currentImu.timestampUs;
    frame.roll = currentImu.roll;
    frame.pitch = currentImu.pitch;
    frame.yaw = currentImu.yaw;
    frame.imuFresh = (currentImu.sampleId != lastPackedSampleId) ? 1 : 0;
    lastPackedSampleId = currentImu.sampleId;
    frame.reserved[0] = 0;
    frame.reserved[1] = 0;
    frame.reserved[2] = 0;

    emgFramesThisWindow++;

    if (targetLocked && WiFi.status() == WL_CONNECTED) {
      packet.frames[packetFrameIndex] = frame;
      packetFrameIndex++;

      if (packetFrameIndex >= EMG_FRAMES_PER_PACKET) {
        packet.header.packetSequence = packetSequenceCounter++;
        queueFrame(packet);
        packetFrameIndex = 0;
      }
    } else {
      packetFrameIndex = 0;
    }

    nextSampleUs += EMG_FRAME_PERIOD_US;
    int32_t remainingUs = static_cast<int32_t>(nextSampleUs - micros());
    if (remainingUs > 0) {
      if (remainingUs > SHORT_BUSY_WAIT_US) {
        delayMicroseconds(static_cast<unsigned int>(remainingUs - SHORT_BUSY_WAIT_US));
      }
      while (static_cast<int32_t>(nextSampleUs - micros()) > 0) {
        // Short final spin for tighter frame timing.
      }
    } else {
      acquisitionOverrunsThisWindow++;
      nextSampleUs = micros();
    }
  }
}

void imuTask(void *parameter) {
  for (;;) {
    if (!imuReady) {
      vTaskDelay(10 / portTICK_PERIOD_MS);
      continue;
    }

    updateLatestImu();
    vTaskDelay(1);
  }
}

void networkTask(void *parameter) {
  EmgImuPacket packet;

  for (;;) {
    handleSerialCommands();
    handleControlPacket();
    monitorWiFiState();
    bool streamActive = targetLocked && WiFi.status() == WL_CONNECTED;

    if (streamActive) {
      uint8_t packetsSentThisPass = 0;
      while (packetsSentThisPass < MAX_UDP_PACKETS_PER_PASS && xQueueReceive(frameQueue, &packet, 0) == pdTRUE) {
        sendFrameUdp(&packet, sizeof(packet));
        packetsSentThisPass++;
      }
    } else {
      flushFrameQueue();
    }

    if (streamActive && (millis() - lastAuthorizedCommandMs) > STREAM_KEEPALIVE_TIMEOUT_MS) {
      Serial.println("Keepalive timed out. Stopping stream.");
      clearStreamTarget();
    }

    printDiagnosticsIfDue();
    vTaskDelay(1);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  deviceId = buildDeviceId();
  deviceName = "BioWave-EMG-IMU-2026";

  Serial.println();
  Serial.println("====================================");
  Serial.println("BioWave 8CH EMG + BNO080 Streamer");
  Serial.print("Device ID: ");
  Serial.println(deviceId);
  Serial.print("Device Name: ");
  Serial.println(deviceName);
  Serial.println("====================================");

  analogReadResolution(12); // Raw ADC values 0..4095
  for (int i = 0; i < EMG_CHANNELS; ++i) {
    pinMode(EMG_PINS[i], INPUT);
    analogSetPinAttenuation(EMG_PINS[i], ADC_11db);
  }

  initImu();

  if (hasSavedWiFiCredentials()) {
    connectToSavedWiFi();
  } else {
    Serial.println("No saved Wi-Fi credentials. Waiting for USB provisioning command.");
  }

  if (!imuReady) {
    Serial.println("IMU failed to start. EMG acquisition will still run, but orientation values will remain zero.");
  }

  frameQueue = xQueueCreate(FRAME_QUEUE_LENGTH, sizeof(EmgImuPacket));
  if (frameQueue == nullptr) {
    Serial.println("FATAL: Failed to create frame queue.");
    while (true) {
      delay(1000);
    }
  }

  diagnosticsWindowStartMs = millis();

  xTaskCreatePinnedToCore(
    acquisitionTask,
    "AcquisitionTask",
    8192,
    nullptr,
    ACQUISITION_PRIORITY,
    &acquisitionTaskHandle,
    ACQUISITION_CORE
  );

  xTaskCreatePinnedToCore(
    imuTask,
    "ImuTask",
    6144,
    nullptr,
    IMU_PRIORITY,
    &imuTaskHandle,
    NETWORK_CORE
  );

  xTaskCreatePinnedToCore(
    networkTask,
    "NetworkTask",
    12288,
    nullptr,
    NETWORK_PRIORITY,
    &networkTaskHandle,
    NETWORK_CORE
  );
}

void loop() {
  // All work runs in the FreeRTOS tasks.
  vTaskDelay(portMAX_DELAY);
}
