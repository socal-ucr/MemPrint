# Out-of-tree Pin tool build. Usage: make PIN_ROOT=/path/to/pin-3.30-...
ifndef PIN_ROOT
$(error PIN_ROOT is not set; run: make PIN_ROOT=/path/to/pin-kit)
endif
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules
