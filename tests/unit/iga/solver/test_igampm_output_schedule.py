from types import SimpleNamespace

from src.igampm.mainIGAMPM import IGAMPM


def test_implicit_facade_records_initial_interval_and_final_frames(
    tmp_path, capsys
):
    recorded_iga_steps = []
    recorded_mpm_steps = []

    iga_engine = SimpleNamespace(
        output_interval=2,
        output_count=0,
        step_count=0,
        time=0.0,
        path=str(tmp_path / "iga"),
    )

    def record_iga(log=True):
        assert log is False
        recorded_iga_steps.append(iga_engine.step_count)
        output = tmp_path / "iga" / f"IGA{iga_engine.output_count:06d}.vtu"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("frame", encoding="utf-8")
        iga_engine.output_count += 1

    iga_engine.record = record_iga

    mpm_engine = SimpleNamespace(
        step_count=0,
        time=0.0,
        path=str(tmp_path / "mpm"),
    )

    def record_mpm(log=True):
        assert log is False
        recorded_mpm_steps.append(mpm_engine.step_count)

    mpm_engine.record = record_mpm

    engine = SimpleNamespace(
        implicit_step_index=0,
        time=0.0,
        activate_fric=False,
    )
    engine._initialize_implicit_ipc_state = lambda: None

    def run_implicit_ipc_contact(steps, postprocessing, **kwargs):
        del kwargs
        for _ in range(steps):
            engine.implicit_step_index += 1
            engine.time += 0.1
            for callback in postprocessing:
                callback(engine)
        return {"completed_steps": steps}

    engine.run_implicit_ipc_contact = run_implicit_ipc_contact

    coupling = object.__new__(IGAMPM)
    coupling.log = False
    coupling.contactor = SimpleNamespace(contact_model="IPC")
    coupling.engine = engine
    coupling.iga_engine = iga_engine
    coupling.mpm_engine = mpm_engine
    coupling.mpm = SimpleNamespace(
        sims=SimpleNamespace(
            current_step=0,
            current_time=0.0,
            current_print=0,
        ),
        first_run=True,
        recorder=None,
        scene=None,
    )
    coupling.iga = SimpleNamespace(
        solver_kwargs={"path": str(tmp_path / "iga")}
    )
    coupling._last_implicit_recorded_step = None
    coupling.build = lambda: engine

    result = coupling.run(steps=3, verbose=False)

    assert result["completed_steps"] == 3
    assert recorded_iga_steps == [0, 2, 3]
    assert recorded_mpm_steps == [0, 2, 3]
    assert (tmp_path / "iga" / "IGA000000.vtu").is_file()
    assert coupling.mpm.sims.current_print == 3
    output = capsys.readouterr().out
    assert output.count("# IGAMPM Save |") == 3
    assert "# IGA Save |" not in output
    assert "# MPM Save |" not in output
