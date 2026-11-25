import os
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
class TestBatchIsendIrecv(unittest.TestCase):
    def setUp(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.comm_ptr_wrapper = torch.tensor(
            [self.comm.py2f()], dtype=torch.int64, device="cpu"
        )

    def _devices(self) -> list[str]:
        return ["cpu", "mcpu"] if MCPU_AVAILABLE else ["cpu"]

    def test_empty_batch(self) -> None:
        torch_mpi_ext.ops.batch_isend_irecv(
            [], [], [], [], [], [], self.comm_ptr_wrapper
        )

    def test_mutual_pair_exchange_multi_dtype(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        dtypes = [
            torch.float32,
            torch.bfloat16,
            torch.uint8,
            torch.float8_e4m3fn,
        ]
        peer = (self.rank + 1) % self.size
        src_peer = (self.rank - 1 + self.size) % self.size
        for device in self._devices():
            with self.subTest(device=device):
                sends = []
                recvs = []
                dst_ranks = []
                src_ranks = []
                send_tags = []
                recv_tags = []
                expected = []
                for idx, dt in enumerate(dtypes):
                    base = torch.arange(16, dtype=torch.float32) + (self.rank + 1) * 10
                    s = base.to(dt).to(device)
                    r = torch.empty((16,), dtype=dt, device=device)
                    sends.append(s)
                    dst_ranks.append(peer)
                    send_tags.append(100 + idx)
                    recvs.append(r)
                    src_ranks.append(src_peer)
                    recv_tags.append(100 + idx)
                    exp_base = (
                        torch.arange(16, dtype=torch.float32) + (src_peer + 1) * 10
                    )
                    expected.append(exp_base.to(dt))

                # Include a 0-element tensor to verify empty payload handling.
                s_empty = torch.empty((0,), dtype=torch.uint8, device=device)
                r_empty = torch.empty((0,), dtype=torch.uint8, device=device)
                sends.append(s_empty)
                dst_ranks.append(peer)
                send_tags.append(999)
                recvs.append(r_empty)
                src_ranks.append(src_peer)
                recv_tags.append(999)

                torch_mpi_ext.ops.batch_isend_irecv(
                    sends,
                    dst_ranks,
                    send_tags,
                    recvs,
                    src_ranks,
                    recv_tags,
                    self.comm_ptr_wrapper,
                )
                _sync_if_needed(device)

                for actual, exp in zip(recvs[:-1], expected):
                    torch.testing.assert_close(
                        actual.cpu().float(), exp.float(), rtol=0, atol=0
                    )

    def test_asymmetric_send_recv_and_subcomm_isolation(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        # Rank 0 sends to all active ranks, other ranks only receive (or vice versa
        # on subcommunicators if size >= 4).
        for device in self._devices():
            with self.subTest(device=device):
                if self.rank == 0:
                    sends = [
                        torch.full((8,), float(r + 1), dtype=torch.bfloat16, device=device)
                        for r in range(1, self.size)
                    ]
                    dsts = list(range(1, self.size))
                    tags = [42 + r for r in range(1, self.size)]
                    torch_mpi_ext.ops.batch_isend_irecv(
                        sends, dsts, tags, [], [], [], self.comm_ptr_wrapper
                    )
                    _sync_if_needed(device)
                else:
                    recv = torch.zeros((8,), dtype=torch.bfloat16, device=device)
                    torch_mpi_ext.ops.batch_isend_irecv(
                        [], [], [], [recv], [0], [42 + self.rank], self.comm_ptr_wrapper
                    )
                    _sync_if_needed(device)
                    torch.testing.assert_close(
                        recv.cpu(),
                        torch.full((8,), float(self.rank + 1), dtype=torch.bfloat16),
                    )

    @unittest.skipIf(not MCPU_AVAILABLE, "mcpu not available")
    def test_custom_mcpu_stream_execution(self) -> None:
        if self.size < 2:
            self.skipTest("requires at least 2 MPI ranks")
        peer = (self.rank ^ 1) if self.size % 2 == 0 or self.rank + 1 < self.size else None
        stream = torch.mcpu.Stream()
        if peer is None:
            torch_mpi_ext.ops.batch_isend_irecv(
                [], [], [], [], [], [], self.comm_ptr_wrapper
            )
            return
        send = torch.full((32,), float(self.rank + 3), dtype=torch.uint8, device="mcpu")
        recv = torch.zeros((32,), dtype=torch.uint8, device="mcpu")
        # Ensure default stream finishes filling `send` before `stream` reads it.
        stream.wait_stream(torch.mcpu.current_stream())
        with torch.mcpu.stream(stream):
            torch_mpi_ext.ops.batch_isend_irecv(
                [send],
                [peer],
                [7],
                [recv],
                [peer],
                [7],
                self.comm_ptr_wrapper,
            )
        # Drop host reference before synchronizing `stream` to exercise
        # cross-stream `recordStream` lifetime without worker Tensor retention.
        del send
        stream.synchronize()
        torch.testing.assert_close(
            recv.cpu(), torch.full((32,), float(peer + 3), dtype=torch.uint8)
        )

    def test_validation_errors(self) -> None:
        t = torch.zeros((4,), dtype=torch.float32, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "same length"):
            torch_mpi_ext.ops.batch_isend_irecv(
                [t], [0, 0], [0], [], [], [], self.comm_ptr_wrapper
            )
        with self.assertRaisesRegex(RuntimeError, "invalid dst rank"):
            torch_mpi_ext.ops.batch_isend_irecv(
                [t], [self.size], [0], [], [], [], self.comm_ptr_wrapper
            )
        with self.assertRaisesRegex(RuntimeError, "invalid MPI send tag"):
            torch_mpi_ext.ops.batch_isend_irecv(
                [t], [0], [-1], [], [], [], self.comm_ptr_wrapper
            )
        if MCPU_AVAILABLE:
            tm = t.to("mcpu")
            with self.assertRaisesRegex(RuntimeError, "same device"):
                torch_mpi_ext.ops.batch_isend_irecv(
                    [t], [0], [0], [tm], [0], [0], self.comm_ptr_wrapper
                )


if __name__ == "__main__":
    unittest.main()
