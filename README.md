# About

To fully utilize the Seplos BMS, one has to use a closed-source, windows-only,
GUI software (Battery Manager).

I didn't really like that, so I wrote a python script that can load/store
settings, and also supports the main status query (code 42).

# Protocol XML

At least in my case the Battery Management software also came with a folder
called `Agreement` (protocol is probably a better name), that has detailed
description of all the message fields (most notably, the settings, but also the
telemetry).

The idea is that the software can support multiple different versions/variants
of the more-or-less same protocol with just an XML file load (before first
use).

Luckily, I can also use the same file to write a simple script to communicate
with any supported BMS, as long as the protocol XML file is present.

# Roadmap

- add support for more commands
  - signal (warnings etc)
  - control (charging on/off)
  - history
- add a web interface, so it can be used with an SBC
- test on other versions/variants, collect more XML files
