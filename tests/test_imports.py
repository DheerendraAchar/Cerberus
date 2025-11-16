def test_package_import():
    import cerberus

    assert hasattr(cerberus, "__version__")
    assert cerberus.hello() == "cerberus scaffold"
