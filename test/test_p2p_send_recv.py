import unittest

import torch

try:
    from mpi4py import MPI

    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

try:
    import torch_mcpu  # noqa: F401

    MCPU_AVAILABLE = True
except ImportError:
    MCPU_AVAILABLE = False

import torch_mpi_ext


def _sync_if_needed(device: str) -> None:
    if device == "mcpu":
        torch.mcpu.synchronize()


@unittest.skipIf(not MPI_AVAILABLE, "mpi4py not available")
class TestP2PSendRecvBroadcast(unittest.TestCase):
    def setUp(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_ptr_wrapper = torch.tensor(
            [self.comm.py2f()], dtype=torch.int64, device="cpu"
        )

    def _devices(self) -> list[str]:
        return ["cpu", "mcpu"] if MCPU_AVAILABLE else ["cpu"]

    def test_send_recv_ring_multi_shape_and_dtype(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        cases = [
            ((16,), torch.bfloat16),
            ((4, 8), torch.float32),
            ((2, 3, 5), torch.int64),
            ((7, 3), torch.int32),
            ((0,), torch.float32),
        ]
        next_rank = (self.rank + 1) % self.size
        prev_rank = (self.rank - 1 + self.size) % self.size
        for device in self._devices():
            with self.subTest(device=device):
                for idx, (shape, dt) in enumerate(cases):
                    numel = 1
                    for dim in shape:
                        numel *= dim
                    if numel > 0:
                        send_tensor = (
                            (torch.arange(numel, dtype=torch.float32) + self.rank * 10 + idx)
                            .reshape(shape)
                            .to(dt)
                            .to(device)
                        )
                        expected = (
                            (torch.arange(numel, dtype=torch.float32) + prev_rank * 10 + idx)
                            .reshape(shape)
                            .to(dt)
                        )
                    else:
                        send_tensor = torch.empty(shape, dtype=dt, device=device)
                        expected = torch.empty(shape, dtype=dt, device="cpu")

                    # Even ranks send first then recv; odd ranks recv first then send.
                    if self.rank % 2 == 0:
                        torch_mpi_ext.ops.send(
                            send_tensor, next_rank, 100 + idx, self.comm_ptr_wrapper
                        )
                        recv_tensor = torch_mpi_ext.ops.recv(
                            shape,
                            dt,
                            prev_rank,
                            100 + idx,
                            self.comm_ptr_wrapper,
                            device=device,
                        )
                    else:
                        recv_tensor = torch_mpi_ext.ops.recv(
                            shape,
                            dt,
                            prev_rank,
                            100 + idx,
                            self.comm_ptr_wrapper,
                            device=device,
                        )
                        torch_mpi_ext.ops.send(
                            send_tensor, next_rank, 100 + idx, self.comm_ptr_wrapper
                        )
                    _sync_if_needed(device)
                    torch.testing.assert_close(
                        recv_tensor.cpu(), expected, rtol=0, atol=0
                    )

    def test_broadcast_inplace_and_out_of_place(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        root = self.size - 1
        for device in self._devices():
            with self.subTest(device=device):
                expected = (
                    torch.arange(24, dtype=torch.float32).reshape(3, 8).to(torch.bfloat16)
                )
                buf = (
                    expected.clone().to(device)
                    if self.rank == root
                    else torch.zeros((3, 8), dtype=torch.bfloat16, device=device)
                )
                ret = torch_mpi_ext.ops.broadcast_(buf, root, self.comm_ptr_wrapper)
                self.assertIs(ret, buf)
                out = torch_mpi_ext.ops.broadcast(buf, root, self.comm_ptr_wrapper)
                _sync_if_needed(device)
                torch.testing.assert_close(buf.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    @unittest.skipIf(not MCPU_AVAILABLE, "mcpu not available")
    def test_cross_stream_send_recv(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        send_stream = torch.mcpu.Stream()
        recv_stream = torch.mcpu.Stream()
        peer = (self.rank + 1) % self.size
        src_peer = (self.rank - 1 + self.size) % self.size

        send_t = torch.full((64,), float(self.rank + 5), dtype=torch.bfloat16, device="mcpu")
        recv_t = torch.empty((64,), dtype=torch.bfloat16, device="mcpu")

        send_stream.wait_stream(torch.mcpu.current_stream())
        with torch.mcpu.stream(send_stream):
            torch_mpi_ext.ops.send(send_t, peer, 77, self.comm_ptr_wrapper)
        with torch.mcpu.stream(recv_stream):
            torch_mpi_ext.ops.recv_(recv_t, src_peer, 77, self.comm_ptr_wrapper)
        del send_t
        send_stream.synchronize()
        recv_stream.synchronize()
        torch.testing.assert_close(
            recv_t.cpu(),
            torch.full((64,), float(src_peer + 5), dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_compiled_send_recv_and_broadcast(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        next_rank = (self.rank + 1) % self.size
        prev_rank = (self.rank - 1 + self.size) % self.size
        root = self.size - 1

        def even_step(
            send_t: torch.Tensor,
            recv_t: torch.Tensor,
            bcast_t: torch.Tensor,
            wrapper: torch.Tensor,
        ) -> torch.Tensor:
            torch_mpi_ext.ops.send(send_t, next_rank, 211, wrapper)
            torch_mpi_ext.ops.recv_(recv_t, prev_rank, 211, wrapper)
            torch_mpi_ext.ops.broadcast_(bcast_t, root, wrapper)
            return recv_t + bcast_t

        def odd_step(
            send_t: torch.Tensor,
            recv_t: torch.Tensor,
            bcast_t: torch.Tensor,
            wrapper: torch.Tensor,
        ) -> torch.Tensor:
            torch_mpi_ext.ops.recv_(recv_t, prev_rank, 211, wrapper)
            torch_mpi_ext.ops.send(send_t, next_rank, 211, wrapper)
            torch_mpi_ext.ops.broadcast_(bcast_t, root, wrapper)
            return recv_t + bcast_t

        for device in self._devices():
            with self.subTest(device=device):
                compile_opts = {"cpp_wrapper": True} if device == "mcpu" else None
                raw_fn = even_step if self.rank % 2 == 0 else odd_step
                step_fn = torch.compile(raw_fn, options=compile_opts)
                send_t = torch.full(
                    (8,), float(self.rank + 1), dtype=torch.float32, device=device
                )
                recv_t = torch.zeros((8,), dtype=torch.float32, device=device)
                bcast_t = (
                    torch.full((8,), 100.0, dtype=torch.float32, device=device)
                    if self.rank == root
                    else torch.zeros((8,), dtype=torch.float32, device=device)
                )
                out = step_fn(send_t, recv_t, bcast_t, self.comm_ptr_wrapper)
                _sync_if_needed(device)
                expected = torch.full((8,), float(prev_rank + 1) + 100.0)
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    def test_validation_errors(self) -> None:
        t = torch.zeros((4, 4), dtype=torch.float32, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            torch_mpi_ext.ops.send(t.t(), 0, 0, self.comm_ptr_wrapper)
        with self.assertRaisesRegex(RuntimeError, "invalid dst rank"):
            torch_mpi_ext.ops.send(t, self.size, 0, self.comm_ptr_wrapper)
        with self.assertRaisesRegex(RuntimeError, "invalid MPI recv tag"):
            torch_mpi_ext.ops.recv_(t, 0, -1, self.comm_ptr_wrapper)
        with self.assertRaisesRegex(RuntimeError, "invalid src rank"):
            torch_mpi_ext.ops.broadcast_(t, self.size, self.comm_ptr_wrapper)


if __name__ == "__main__":
    unittest.main()
