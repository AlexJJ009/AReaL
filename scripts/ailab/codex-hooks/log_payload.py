#!/usr/bin/env python3
# Shim: sessions started before guard.py replaced it still call this path; allow and exit.
import sys

sys.stdin.read()
