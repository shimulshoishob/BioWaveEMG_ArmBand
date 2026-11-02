// Copy this file to secrets.h (which is gitignored) and set a long random key.
// The same key must be entered in the desktop app (code/device_secret.json).
// Generate one with:  python3 -c "import secrets; print(secrets.token_hex(24))"
#pragma once
#define DEVICE_ACCESS_KEY_VALUE "CHANGE_THIS_TO_A_LONG_RANDOM_KEY"
