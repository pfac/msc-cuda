CMAKE_BUILD_TYPES = None Debug Release RelWithDebInfo MinSizeRel

BUILD_DIR ?= build
BUILD_MAKEFILE ?= Makefile.$(BUILD_DIR)

.PHONY: default $(CMAKE_BUILD_TYPES) clean purge reset

default: None

$(CMAKE_BUILD_TYPES): $(BUILD_DIR) $(BUILD_DIR)/Makefile
		@echo "  -> $< [$@]"
		@cd $<; $_ $@

$(BUILD_DIR):
		@echo "  (MKDIR) $@"
		@mkdir -p $@

$(BUILD_DIR)/Makefile: $(BUILD_MAKEFILE) $(BUILD_DIR)
		@echo "  (COPY) $< => $@"
		@cp $< $@

clean:
		@echo "  (CLEANING)"
		@if [ -d $(BUILD_DIR) ]; then cd $(BUILD_DIR); $_ clean; fi;

purge:
		@echo "  (PURGING)"
		@$(RM) -r $(BUILD_DIR)

reset: purge default
