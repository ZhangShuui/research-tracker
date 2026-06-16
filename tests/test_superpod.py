"""Tests for the SuperPod SSH channel (command construction; no live connection)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from paper_tracker import superpod


def test_run_uses_controlmaster_and_login_shell():
    with patch("paper_tracker.superpod.subprocess.run") as m:
        m.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        superpod.run("squeue -u $USER")
    argv = m.call_args.args[0]
    assert argv[0] == "ssh"
    assert "ControlMaster=auto" in argv
    assert any(a.startswith("ControlPath=") and "pt-" in a for a in argv)  # app-owned socket
    assert "BatchMode=yes" in argv
    assert argv[-2] == "superpod"
    assert argv[-1].startswith("bash -lc ")            # login-shell wrapped
    assert "squeue -u $USER" in argv[-1]


def test_run_no_login_shell_passes_raw():
    with patch("paper_tracker.superpod.subprocess.run") as m:
        m.return_value = MagicMock(returncode=0)
        superpod.run("true", login_shell=False)
    assert m.call_args.args[0][-1] == "true"


def test_slurm_loads_module_first():
    with patch("paper_tracker.superpod.run") as m:
        superpod.slurm("squeue")
    assert "module load slurm && squeue" in m.call_args.args[0]


def test_master_alive_checks_control_socket():
    with patch("paper_tracker.superpod.subprocess.run") as m:
        m.return_value = MagicMock(returncode=0)
        assert superpod.master_alive() is True
    argv = m.call_args.args[0]
    assert argv[:3] == ["ssh", "-O", "check"]


def test_reachable_false_on_exception():
    with patch("paper_tracker.superpod.run", side_effect=Exception("vpn down")):
        assert superpod.reachable() is False


# --- SOP primitives ---

def test_make_workdir_slugifies_and_honors_override():
    with patch("paper_tracker.superpod.run") as m:
        m.return_value = MagicMock(returncode=0, stdout="/custom/root/repro-foo-bar\n", stderr="")
        out = superpod.make_workdir("Foo Bar!!", project_root="/custom/root")
    assert out == "/custom/root/repro-foo-bar"
    assert "/custom/root/repro-foo-bar" in m.call_args.args[0]   # per-call override used


def test_make_workdir_uses_config_project_root():
    with patch("paper_tracker.superpod._conf", return_value={"project_root": "/data/myrepro"}), \
         patch("paper_tracker.superpod.run") as m:
        m.return_value = MagicMock(returncode=0, stdout="/data/myrepro/repro-x\n", stderr="")
        superpod.make_workdir("x")
    assert "/data/myrepro/repro-x" in m.call_args.args[0]        # config-driven root


def test_host_and_root_honor_config_with_fallback():
    with patch("paper_tracker.superpod._conf", return_value={"host": "mycluster"}):
        assert superpod._host() == "mycluster"
        assert superpod._project_root() == "/project/visworld01"  # falls back to default
    with patch("paper_tracker.superpod._conf", return_value={}):
        assert superpod._host() == "superpod"


def test_write_remote_pipes_stdin():
    with patch("paper_tracker.superpod.run") as m:
        m.return_value = MagicMock(returncode=0, stderr="")
        superpod.write_remote("/w/run.sbatch", "#!/bin/bash\necho hi\n")
    assert "cat > " in m.call_args.args[0]
    assert m.call_args.kwargs.get("input") == "#!/bin/bash\necho hi\n"


def test_submit_parses_jobid():
    with patch("paper_tracker.superpod.slurm") as m:
        m.return_value = MagicMock(returncode=0, stdout="Submitted batch job 1234567\n", stderr="")
        assert superpod.submit("/w/run.sbatch", workdir="/w") == "1234567"


def test_submit_raises_without_jobid():
    import pytest
    with patch("paper_tracker.superpod.slurm") as m:
        m.return_value = MagicMock(returncode=1, stdout="", stderr="sbatch: error")
        with pytest.raises(RuntimeError):
            superpod.submit("/w/run.sbatch")


def test_job_state_parses_sacct():
    with patch("paper_tracker.superpod.slurm") as m:
        m.return_value = MagicMock(returncode=0, stdout="  RUNNING             \n", stderr="")
        assert superpod.job_state("123") == "RUNNING"


def test_wait_polls_until_terminal():
    with patch("paper_tracker.superpod.job_state", side_effect=["PENDING", "RUNNING", "COMPLETED"]), \
         patch("paper_tracker.superpod.time.sleep", lambda *_a, **_k: None):
        assert superpod.wait("123", poll=1, timeout=60) == "COMPLETED"
