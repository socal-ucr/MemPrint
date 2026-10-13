# rw_trace.so, an NVBit tool. NVBIT_PATH: the core/ directory of an NVBit release (nvbit.h, libnvbit.a,
# utils/). Same build as NVBit's tools/mem_trace.
NVBIT_PATH ?= ../../nvbit_release/core
ARCH ?= all
NVCC = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
INCLUDES = -I$(NVBIT_PATH)
LIBS = -L$(NVBIT_PATH) -lnvbit
NVCC_PATH = -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

all: rw_trace.so

rw_trace.so: rw_trace.o inject_funcs.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) -O3 rw_trace.o inject_funcs.o $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

rw_trace.o: rw_trace.cu common.h
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -arch=$(ARCH) -O3 \
		-Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu common.h
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) \
		-Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o
