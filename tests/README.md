## Summary of pytest features for my future reference
1. **@pytest.mark.<markername>**: use this with pytest -m markername to only run the marked tests. Or use the builtin markers like skip (always skip), parameterize (multiple calls to same function).
2. **@pytest.fixture**: Baseline environment or common setup/configs across tests. Decorate a function, class or a module and then pass it as an **argument to test functions**.
3. **conftest.py**: Module level fixture. pytest first looks for within file fixtures and if not found, it tries to resolve the fixture param from conftest.py.
4. **@pytest.mark.parametrize**: @pytest.mark.parametrize(("param1", "param2", "param3"), [(value_a1, value_a2, value_a3), (value_b1, value_b2, value_b3)]). Tuple of param names in quotes followed by list of tuples of param values in order of param tuple.

> Cheatsheet for later reference: https://github.com/mananrg/Pytest-Cheatsheet
