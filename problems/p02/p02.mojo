from memory import UnsafePointer
from gpu import thread_idx, DeviceBuffer
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: add
comptime SIZE = 4
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = SIZE
comptime dtype = DType.float32

fn add(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    input_b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    i = thread_idx.x
    output[i] = input_a[i] + input_b[i]

def run_add(
    context: DeviceContext,
    output_buffer: DeviceBuffer[DType.float32], 
    input_a_buffer: DeviceBuffer[DType.float32], 
    input_b_buffer: DeviceBuffer[DType.float32]
):
    context.enqueue_function[add, add](
        output_buffer,
        input_a_buffer,
        input_b_buffer,
        grid_dim=BLOCKS_PER_GRID,
        block_dim=THREADS_PER_BLOCK,
    )

# ANCHOR_END: add


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE)
        out.enqueue_fill(0)
        input_a = ctx.enqueue_create_buffer[dtype](SIZE)
        input_a.enqueue_fill(0)
        input_b = ctx.enqueue_create_buffer[dtype](SIZE)
        input_b.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)
        with input_a.map_to_host() as a_host, input_b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = Float32(i)
                b_host[i] = Float32(i)
                expected[i] = Float32(a_host[i] + b_host[i])

        run_add(
            context=ctx,
            output_buffer=out,
            input_a_buffer=input_a,
            input_b_buffer=input_b,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_host[i], expected[i])
