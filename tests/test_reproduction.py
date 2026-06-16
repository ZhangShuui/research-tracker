"""Tests for the 4-stage reproduction orchestration (mocked; no live SuperPod)."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from paper_tracker import reproduction


def _ok(stdout="", returncode=0):
    return MagicMock(returncode=returncode, stdout=stdout, stderr="")


# --- Stage 1: init -------------------------------------------------------------
def test_init_makes_workdir_and_clones():
    run = MagicMock(return_value=_ok())
    mw = MagicMock(return_value="/p/repro-fp")
    with patch.multiple("paper_tracker.reproduction.superpod", make_workdir=mw, run=run):
        repro = reproduction.init("fp", repo_url="https://h/r.git")
    assert repro.workdir == "/p/repro-fp"
    mw.assert_called_once()
    cmds = [c.args[0] for c in run.call_args_list]
    assert any("mkdir -p" in c and "logs" in c and "outputs" in c for c in cmds)
    assert any("git clone" in c and "r.git" in c for c in cmds)


def test_init_no_repo_skips_clone():
    run = MagicMock(return_value=_ok())
    with patch.multiple("paper_tracker.reproduction.superpod",
                        make_workdir=MagicMock(return_value="/p/repro-fp"), run=run):
        reproduction.init("fp")
    assert not any("git clone" in c.args[0] for c in run.call_args_list)


# --- Stage 2: build_container --------------------------------------------------
def test_build_container_submits_and_records_job():
    repro = reproduction.Reproduction(name="framepack", workdir="/p/repro-framepack")
    submit = MagicMock(return_value="999")
    wait = MagicMock(return_value="COMPLETED")
    wr = MagicMock()
    fetch = MagicMock(return_value="BUILD_OK")
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=wr, submit=submit, wait=wait, fetch=fetch):
        state, log = reproduction.build_container(
            repro, base="docker://nvcr.io#nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
            build_script="echo build")
    assert state == "COMPLETED" and log == "BUILD_OK"
    assert repro.jobs["build"] == {"job_id": "999", "state": "COMPLETED"}
    written = [c.args[0] for c in wr.call_args_list]
    assert any(p.endswith("_build.sh") for p in written)
    assert any(p.endswith("_build.sbatch") for p in written)
    fetch.assert_called_with("/p/repro-framepack/logs/framepack-build-999.out")


def test_build_container_atomic_save_and_image_ref():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    wr = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=wr, submit=MagicMock(return_value="1"),
                        wait=MagicMock(return_value="COMPLETED"),
                        fetch=MagicMock(return_value="")):
        reproduction.build_container(repro, base="docker://nvcr.io#x", build_script="echo x")
    sbatch = next(c.args[1] for c in wr.call_args_list if c.args[0].endswith("_build.sbatch"))
    assert "docker://nvcr.io#x" in sbatch
    assert "/p/repro-fp/fp.sqsh.tmp" in sbatch  # saves to tmp first
    assert "mv -f /p/repro-fp/fp.sqsh.tmp /p/repro-fp/fp.sqsh" in sbatch  # atomic swap
    assert "unset http_proxy" in sbatch  # proxy hard-unset


def test_build_container_dry_run_does_not_submit():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    submit = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=MagicMock(), submit=submit, wait=MagicMock()):
        state, text = reproduction.build_container(
            repro, base="docker://nvcr.io#x", build_script="echo x", dry_run=True)
    assert state == "DRY_RUN"
    assert "#SBATCH" in text and "container-save" in text
    submit.assert_not_called()


def test_build_container_patch_uses_local_sqsh_base():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    wr = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=wr, submit=MagicMock(return_value="2"),
                        wait=MagicMock(return_value="COMPLETED"), fetch=MagicMock(return_value="")):
        reproduction.build_container(repro, base="/p/repro-fp/fp.sqsh",
                                     build_script="pip install x", save_as="/p/repro-fp/fp.sqsh")
    sbatch = next(c.args[1] for c in wr.call_args_list if c.args[0].endswith("_build.sbatch"))
    assert "--container-image /p/repro-fp/fp.sqsh" in sbatch


# --- Stage 3: run_step ---------------------------------------------------------
def test_run_step_in_container():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    wr = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=wr, submit=MagicMock(return_value="5"),
                        wait=MagicMock(return_value="COMPLETED"), fetch=MagicMock(return_value="done")):
        state, log = reproduction.run_step(
            repro, "infer", "python3.10 x.py", container="/p/repro-fp/fp.sqsh",
            workdir="/p/repro-fp/Repo")
    assert state == "COMPLETED" and repro.jobs["infer"]["job_id"] == "5"
    sbatch = next(c.args[1] for c in wr.call_args_list if c.args[0].endswith("_infer.sbatch"))
    assert "--container-image /p/repro-fp/fp.sqsh" in sbatch
    assert "--container-workdir /p/repro-fp/Repo" in sbatch
    assert "python3.10 x.py" in sbatch


def test_run_step_on_host_has_no_container():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    wr = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=wr, submit=MagicMock(return_value="6"),
                        wait=MagicMock(return_value="COMPLETED"), fetch=MagicMock(return_value="")):
        reproduction.run_step(repro, "download", "python3.10 dl.py")
    sbatch = next(c.args[1] for c in wr.call_args_list if c.args[0].endswith("_download.sbatch"))
    assert "--container-image" not in sbatch
    assert "cd /p/repro-fp && python3.10 dl.py" in sbatch


def test_run_step_dry_run():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    submit = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod",
                        write_remote=MagicMock(), submit=submit, wait=MagicMock()):
        state, text = reproduction.run_step(repro, "infer", "python x.py", dry_run=True)
    assert state == "DRY_RUN" and "#SBATCH" in text
    submit.assert_not_called()


# --- Stage 4: collect / summarize / write_report -------------------------------
def test_collect_handles_missing_files():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")

    def fake_fetch(path, **k):
        if "good.json" in path:
            return "{}"
        raise RuntimeError("nope")

    with patch.multiple("paper_tracker.reproduction.superpod",
                        fetch=MagicMock(side_effect=fake_fetch)):
        out = reproduction.collect(repro, files=("outputs/good.json", "outputs/bad.json"))
    assert out["outputs/good.json"] == "{}"
    assert "fetch failed" in out["outputs/bad.json"]


def test_summarize_calls_llm_with_results():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    with patch("paper_tracker.reproduction.call_cli", return_value="# Report") as cc:
        out = reproduction.summarize(repro, cfg={"summarizer": {}}, paper="FramePack",
                                     collected={"metrics.json": '{"a":1}'}, notes="constant VRAM")
    assert out == "# Report"
    prompt = cc.call_args.args[0]
    assert "FramePack" in prompt and "metrics.json" in prompt and "constant VRAM" in prompt
    assert cc.call_args.kwargs.get("model") == "opus"


def test_write_report():
    repro = reproduction.Reproduction(name="fp", workdir="/p/repro-fp")
    wr = MagicMock()
    with patch.multiple("paper_tracker.reproduction.superpod", write_remote=wr):
        path = reproduction.write_report(repro, "# hi")
    assert path == "/p/repro-fp/report.md"
    wr.assert_called_once_with("/p/repro-fp/report.md", "# hi")
