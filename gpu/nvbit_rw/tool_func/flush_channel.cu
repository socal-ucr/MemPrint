/* Pushes what is left in the device side of the channel to the host. Loaded by rw_trace.cu as a tool
 * module (built to a fatbin and embedded with bin2c, as in NVBit's tools/mem_trace). */
#include "utils/channel.hpp"

extern "C" __global__ void flush_channel(ChannelDev *ch_dev) { ch_dev->flush(); }
