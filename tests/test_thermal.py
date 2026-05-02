from mtplx import thermal


def test_detect_thermal_control_reports_none_without_tools(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    detected = thermal.detect_thermal_control()

    assert detected["available"] is False
    assert detected["selected"] is None
    assert "Install ThermalForge" in detected["instructions"]
    thermal.detect_thermal_control.cache_clear()


def test_set_thermal_profile_without_tool_is_actionable(monkeypatch):
    thermal.detect_thermal_control.cache_clear()
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    result = thermal.set_thermal_profile("performance")

    assert result["ok"] is False
    assert result["profile"] == "performance"
    assert "Install ThermalForge" in result["message"]
    thermal.detect_thermal_control.cache_clear()


def test_thermalforge_profile_candidates_are_explicit():
    commands = thermal._profile_command_candidates(
        {"kind": "thermalforge", "path": "/usr/local/bin/thermalforge"},
        "max",
    )

    assert commands[0] == ["/usr/local/bin/thermalforge", "profile", "Max"]
