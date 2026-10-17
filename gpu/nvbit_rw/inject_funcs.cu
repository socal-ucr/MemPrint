/* Device code NVBit inserts before every global or generic memory instruction (rw_trace.cu). */
#include <stdint.h>
#include <stdio.h>

#include "utils/channel.hpp"
#include "utils/utils.h"

#include "common.h"

extern "C" __device__ __noinline__ void instrument_mem(int pred, int flags, int size, uint64_t addr,
                                                       uint64_t event, uint64_t pchannel_dev) {
    if (!pred) return;
    const int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;
    mem_access_t ma;
    for (int i = 0; i < 32; i++) ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    ma.event = event;
    ma.active_mask = active_mask;
    ma.flags = flags;
    ma.size = size;
    ma.cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if (first_laneid == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}
