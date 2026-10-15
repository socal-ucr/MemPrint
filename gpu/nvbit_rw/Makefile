# rw_trace.so, an NVBit tool. NVBIT_PATH: the core/ directory of an NVBit release (nvbit.h, libnvbit.a,
# utils/). Same build as NVBit's tools/mem_trace (1.8): the flush_channel kernel is built to a fatbin and
# embedded as a C array, which rw_trace.cu loads at run time as a tool module.
NVBIT_PATH ?= ../../nvbit_release/core
ARCH ?= all
NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
INCLUDES = -I$(NVBIT_PATH)
LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))
# ptxas 12.3 and later choose the register count of the injected function themselves (as mem_trace does)
PTXAS_VER = $(shell ptxas --version | grep release | cut -f2 -d, | cut -f3 -d' ')
MAXRREGCOUNT = $(if $(filter 0,$(shell echo "$(PTXAS_VER) >= 12.3" | bc)),-maxrregcount=24,)

all: rw_trace.so

rw_trace.so: rw_trace.o inject_funcs.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 rw_trace.o inject_funcs.o $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

tool_func/flush_channel.fatbin: tool_func/flush_channel.cu
	$(NVCC) -arch=$(ARCH) $(INCLUDES) -O3 -fatbin $< -o $@

tool_func/flush_channel.c: tool_func/flush_channel.fatbin
	bin2c -c --name flush_channel_bin $< > $@

rw_trace.o: rw_trace.cu common.h tool_func/flush_channel.c
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) -O3 \
		-Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT) -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) \
		-Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o tool_func/*.fatbin tool_func/*.c
