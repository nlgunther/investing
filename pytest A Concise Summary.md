# pytest: A Concise Summary

**pytest** is a powerful, flexible testing framework for Python that makes it easy to write simple unit tests and complex functional tests. It supports test discovery, fixtures for setup/teardown, parametrization, plugins, and more — all with minimal boilerplate.

---

## Key Features

- **No API required**: Write tests as plain functions (no class inheritance needed).
- **Automatic test discovery**: Finds tests in files named `test_*.py` or `*test.py`.
- **Rich assertions**: Uses Python’s `assert` with detailed failure reports.
- **Fixtures**: Reusable setup/teardown logic.
- **Parametrization**: Run the same test with multiple inputs.
- **Extensible via plugins**: `pytest-cov`, `pytest-django`, etc.

Install with:

```bash
pip install pytest
```

Run tests:

```bash
pytest
```

---

## Basic Test Example

```python
# test_basic.py
def add(a, b):
    return a + b

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0
```

Run:

```bash
pytest -q
..
2 passed
```

---

## Fixtures: Reusable Test Context

Fixtures provide setup and teardown logic. Define them with `@pytest.fixture`.

### Simple Fixture

```python
# test_fixture.py
import pytest

@pytest.fixture
def sample_data():
    return [1, 2, 3, 4]

def test_sum(sample_data):
    assert sum(sample_data) == 10

def test_len(sample_data):
    assert len(sample_data) == 4
```

The `sample_data` fixture is injected into tests that declare it as a parameter.

---

## Fixture Scope Management

Control how often a fixture is set up and torn down using the `scope` parameter.

| Scope        | Execution Frequency         |
| ------------ | --------------------------- |
| `"function"` | Per test function (default) |
| `"class"`    | Once per test class         |
| `"module"`   | Once per module             |
| `"session"`  | Once per test session       |

### Example: Different Scopes

```python
# test_scopes.py
import pytest

@pytest.fixture(scope="session")
def db_connection():
    print("\nConnecting to DB (session)")
    conn = "DB_CONN"
    yield conn
    print("Closing DB (session)")

@pytest.fixture(scope="module")
def api_client():
    print("  Creating API client (module)")
    client = "API_CLIENT"
    yield client
    print("  Destroying API client (module)")

@pytest.fixture(scope="function")
def user(api_client):
    print("    Creating user (function)")
    user = f"User_{id(user)}"
    yield user
    print("    Deleting user (function)")

def test_user_login(user, db_connection):
    assert "User_" in user

def test_user_profile(user, db_connection):
    assert user.startswith("User_")
```

**Output when running:**

```
Connecting to DB (session)
  Creating API client (module)
    Creating user (function)
.    Deleting user (function)
    Creating user (function)
.    Deleting user (function)
  Destroying API client (module)
Closing DB (session)
```

> Use `session` for expensive resources (e.g., DB, browser), `module` for shared module state, `function` for isolated tests.

---

## Advanced Fixture: `request` and Parametrization

```python
@pytest.fixture(scope="function", params=["chrome", "firefox"])
def browser(request):
    b = request.param
    print(f"\nStarting {b}")
    yield b
    print(f"Closing {b}")

def test_browser_title(browser):
    assert browser in ["chrome", "firefox"]
```

This runs the test twice — once with `"chrome"`, once with `"firefox"`.

---

## Summary Table: Fixture Scopes

| Scope      | Setup Once Per     | Use Case               |
| ---------- | ------------------ | ---------------------- |
| `function` | Each test function | Isolated test data     |
| `class`    | Each test class    | Shared class state     |
| `module`   | Each `.py` file    | Module-level resources |
| `session`  | Entire test run    | DB, server, config     |

---

## Best Practices

- Keep tests small and fast.
- Use descriptive test names: `test_function_expected_behavior`.
- Use fixtures instead of manual setup/teardown.
- Prefer `session` or `module` scope for performance.
- Use `tmp_path` (built-in) for temp files.

```python
def test_write_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("hello")
    assert file.read_text() == "hello"
```

---

**pytest** is the de facto standard for Python testing due to its simplicity, power, and ecosystem. Master fixtures and scope — and you’ll write clean, efficient, maintainable tests.



# Parametrized Tests in **pytest** — In-Depth Guide

**Parametrization** allows you to **run the same test with multiple sets of inputs and expected outputs**, eliminating repetitive code and improving test coverage.

---

## Why Parametrize?

Instead of writing:

```python
def test_add_1():
    assert add(1, 2) == 3

def test_add_2():
    assert add(1, -1) == 0

def test_add_3():
    assert add(0, 0) == 0
```

You write **one** test that runs **three times**:

```python
@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (1, -1, 0),
    (0, 0, 0),
])
def test_add(a, b, expected):
    assert add(a, b) == expected
```

---

## Basic Syntax

```python
@pytest.mark.parametrize(argnames, argvalues)
```

- `argnames`: Comma-separated string of parameter names.
- `argvalues`: List of values (tuples if multiple args).

### Example: Simple Values

```python
import pytest

@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_is_positive(number):
    assert number > 0
```

Runs 4 times: `number = 1`, `2`, `3`, `4`.

---

## Multiple Parameters

```python
@pytest.mark.parametrize("x, y, result", [
    (1, 1, 2),
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
])
def test_add(x, y, result):
    assert x + y == result
```

Each tuple `(x, y, result)` becomes one test case.

---

## Using `pytest.param()` for Custom IDs & Marks

```python
@pytest.mark.parametrize("x, y, expected", [
    pytest.param(1, 1, 2, id="1+1"),
    pytest.param(10, 20, 30, id="10+20"),
    pytest.param(-1, 1, 0, id="negative"),
    pytest.param(0, 0, 0, marks=pytest.mark.xfail, id="zero"),
], ids=["a", "b"])  # overrides if both used
def test_add(x, y, expected):
    assert x + y == expected
```

### Output:

```
test_add[1+1] PASSED
test_add[10+20] PASSED
test_add[negative] PASSED
test_add[zero] XFAIL
```

> Use `id=` for readable test names in reports.

---

## Parametrizing with Fixtures

You can **combine fixtures and parametrization**.

```python
@pytest.fixture
def user_role():
    return "admin"

@pytest.mark.parametrize("username, expected_access", [
    ("alice", True),
    ("bob", False),
])
def test_user_access(username, expected_access, user_role):
    user = User(username, role=user_role)
    assert user.has_access() == expected_access
```

> `user_role` is injected **once per test case**, alongside parametrized args.

---

## Multiple Parametrizations (Cartesian Product)

Use **multiple `@pytest.mark.parametrize`** to test all combinations.

```python
@pytest.mark.parametrize("color", ["red", "blue"])
@pytest.mark.parametrize("size", ["small", "large"])
def test_shirt(color, size):
    shirt = Shirt(color, size)
    assert shirt.is_valid()
```

Runs **4 tests**:

- `color=red, size=small`
- `color=red, size=large`
- `color=blue, size=small`
- `color=blue, size=large`

> Order matters: outer decorator = slower-varying parameter.

---

## Parametrizing at Class Level

```python
@pytest.mark.parametrize("value", [1, 2, 3])
class TestMath:
    def test_even(self, value):
        assert value % 2 == value & 1  # bit trick

    def test_positive(self, value):
        assert value > 0
```

All methods in the class run for each `value`.

---

## Advanced: Indirect Parametrization

Use `indirect=True` to **pass parameters to fixtures**, not the test directly.

### Use Case: Setup different test environments

```python
import pytest

@pytest.fixture
def setup_db(request):
    db_type = request.param
    if db_type == "sqlite":
        db = SQLiteDB()
    elif db_type == "postgres":
        db = PostgresDB()
    db.create()
    yield db
    db.destroy()

@pytest.mark.parametrize("setup_db", ["sqlite", "postgres"], indirect=True)
def test_insert(setup_db):
    setup_db.insert("data")
    assert setup_db.count() == 1
```

- `setup_db` fixture receives `"sqlite"` or `"postgres"` as `request.param`.
- Test runs twice with different DB backends.

---

## Parametrization with `ids` for Readability

```python
@pytest.mark.parametrize(
    "input, expected",
    [(1, 2), (3, 6), (5, 10)],
    ids=["1→2", "3→6", "5→10"]
)
def test_double(input, expected):
    assert double(input) == expected
```

Or use a function:

```python
def idfn(val):
    if isinstance(val, dict):
        return f"user:{val['name']}"
    return str(val)

@pytest.mark.parametrize("user", [
    {"name": "alice", "role": "admin"},
    {"name": "bob", "role": "user"},
], ids=idfn)
def test_role(user):
    assert user["role"] in ["admin", "user"]
```

---

## Skipping or Xfail in Parametrized Tests

```python
@pytest.mark.parametrize("x, y, expected", [
    (1, 1, 2),
    pytest.param(2, 2, 5, marks=pytest.mark.xfail(reason="bug #123")),
    pytest.param(0, 0, 0, marks=pytest.mark.skipif(sys.platform == "win32", reason="no div0 on win")),
])
def test_add(x, y, expected):
    assert x + y == expected
```

---

## Best Practices

| Do                             | Don't                      |
| ------------------------------ | -------------------------- |
| Use meaningful `ids`           | Leave default `[0]`, `[1]` |
| Keep parametrization small     | Avoid 1000+ cases          |
| Combine with fixtures          | Repeat setup code          |
| Use `indirect` for setup logic | Overuse it                 |

---

## Real-World Example: API Testing

```python
import pytest
import requests

@pytest.mark.parametrize("endpoint, status_code", [
    ("/health", 200),
    ("/users", 200),
    ("/admin", 403),
    pytest.param("/crash", 500, marks=pytest.mark.xfail),
], ids=lambda x: x.upper() if isinstance(x, str) else None)
def test_api_endpoints(endpoint, status_code):
    response = requests.get(f"https://api.example.com{endpoint}")
    assert response.status_code == status_code
```

---

## Summary Table

| Feature                   | Syntax                                            |
| ------------------------- | ------------------------------------------------- |
| Basic                     | `@pytest.mark.parametrize("x", [1,2,3])`          |
| Multiple args             | `@pytest.mark.parametrize("x,y", [(1,2), (3,4)])` |
| Custom IDs                | `id="case1"` or `ids=[...]`                       |
| Skip/Xfail per case       | `pytest.param(..., marks=...)`                    |
| Indirect (fixture params) | `indirect=True`                                   |
| Cartesian product         | Multiple decorators                               |

---

**Parametrized tests = DRY, readable, powerful.**  
Master them — and your test suite becomes **concise, maintainable, and thorough**.

# **pytest Plugins for Parametrization** — Advanced & Scalable Testing

While **built-in `@pytest.mark.parametrize`** is powerful, **plugins** extend it dramatically — enabling **data-driven testing**, **external data sources**, **hypothesis-based property testing**, and **parallel execution**.

Here’s a curated guide to the **best and most useful pytest parametrization plugins**.

---

## 1. `pytest-parametrization` (Built-in — Just a Reminder)

> Not a plugin — **core feature**, but worth anchoring.

```python
@pytest.mark.parametrize("x,y", [(1,2), (3,4)])
def test_add(x, y): ...
```

All plugins below **build on or replace** this.

---

## 2. `pytest-cases` — **Structured, Reusable, Composable Parametrization**

[https://github.com/smarie/python-pytest-cases](https://github.com/smarie/python-pytest-cases)

### Why Use It?

- Avoid **Cartesian explosion** with multiple `@parametrize`.
- **Separate test cases from test logic**.
- **Reusable case functions/fixtures**.
- **Cleaner, more maintainable**.

### Install

```bash
pip install pytest-cases
```

---

### Example: Clean Case Separation

```python
# cases.py
from pytest_cases import case, parametrize_with_cases

@case(tags=["fast"])
def case_add_positive():
    return 2, 3, 5

@case(tags=["edge"])
def case_add_zero():
    return 0, 0, 0

@case(id="neg")
def case_add_negative():
    return -1, 1, 0

# test_math.py
@parametrize_with_cases("a,b,expected", cases=".", prefix="case_")
def test_add(a, b, expected):
    assert a + b == expected
```

**Output:**

```
test_add[case_add_positive-fast] PASSED
test_add[case_add_zero-edge] PASSED
test_add[neg] PASSED
```

---

### Advanced: Fixture + Case Composition

```python
@case
def case_user_admin():
    user = User("admin")
    return user, True

@case
def case_user_guest():
    user = User("guest")
    return user, False

@parametrize_with_cases("user, can_edit", cases=".")
def test_edit_permission(user, can_edit):
    assert user.can_edit() == can_edit
```

---

### Features

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| `@case`             | Define reusable test cases    |
| `tags`              | Filter with `-m "fast"`       |
| `prefix`            | Auto-discover cases           |
| `has_tag()`         | Conditional logic in cases    |
| `@parametrize_plus` | Enhanced built-in parametrize |

---

## 3. `pytest-param-files` — **Load Params from YAML/JSON/CSV**

[https://github.com/chbndrhnns/pytest-param-files](https://github.com/chbndrhnns/pytest-param-files)

### Use Case: **Data-Driven Testing with External Files**

```yaml
# data/login.yaml
- username: admin
  password: secret
  success: true
- username: guest
  password: wrong
  success: false
```

```python
# test_login.py
from pytest_param_files import param_files

@param_files("data/login.yaml")
def test_login(username, password, success):
    assert login(username, password) == success
```

Supports: `.yaml`, `.json`, `.csv`, `.toml`

---

## 4. `pytest-datadir` + `pytest-param-files` — Shared Test Data

```python
# test_data.py
def test_process(datadir):
    data = (datadir / "input.csv").read_text()
    assert process(data) == expected
```

---

## 5. **Hypothesis** — **Property-Based Testing** (Parametrization on Steroids)

[https://hypothesis.works](https://hypothesis.works)

### Install

```bash
pip install hypothesis
```

### Core Idea:

> Instead of **fixed examples**, define **properties** that must hold for **any valid input**.

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers())
def test_add_commutative(x, y):
    assert add(x, y) == add(y, x)
```

**Runs 100+ random tests automatically** — finds edge cases you didn’t think of.

---

### Advanced Strategies

```python
@given(
    st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=10),
    st.sampled_from(["mean", "median", "sum"])
)
def test_stats(numbers, func_name):
    result = stats(numbers, func_name)
    assert isinstance(result, (int, float))
```

---

### Shrinking

When a test fails, Hypothesis **shrinks** input to the **simplest failing case**.

```
Falsified on: numbers=[0], func_name='mean'
```

---

## 6. `pytest-lazy-fixture` — **Dynamic Parametrization with Fixtures**

[https://github.com/pytest-dev/pytest-lazy-fixture](https://github.com/pytest-dev/pytest-lazy-fixture)

### Problem:

You can’t do:

```python
@pytest.mark.parametrize("fix", [db_fixture, api_fixture])  # Error
```

### Solution: `lazy_fixture()`

```python
from pytest_lazyfixture import lazy_fixture

@pytest.mark.parametrize("client", [
    lazy_fixture("db_client"),
    lazy_fixture("api_client"),
    lazy_fixture("mock_client"),
])
def test_connect(client):
    assert client.connect() is True
```

---

## 7. `pytest-parametrize-plus` — **Enhanced Built-in Parametrize**

Part of `pytest-cases`, but can be used standalone.

```python
from pytest_cases import parametrize_plus

@parametrize_plus("x,y", [(1,2), (3,4)], ids=lambda x,y: f"{x}+{y}")
def test_add(x, y):
    ...
```

Supports:

- `ids` as callable with all args
- `filter` to skip cases
- `scope="module"` on parametrize

---

## 8. `pytest-subtests` — **Subtests Inside Parametrized Tests**

[https://github.com/pytest-dev/pytest-subtests](https://github.com/pytest-dev/pytest-subtests)

```python
def test_complex(subtests):
    for i in range(5):
        with subtests.test(i=i):
            assert process(i) == i * 2
```

**One failure doesn’t stop others** — great for loops.

---

## Comparison Table

| Plugin                | Best For            | External Data | Reusability | Property Testing | Dynamic Fixtures |
| --------------------- | ------------------- | ------------- | ----------- | ---------------- | ---------------- |
| `pytest-cases`        | Clean case logic    | No            | Yes         | No               | Yes              |
| `pytest-param-files`  | YAML/CSV tests      | Yes           | No          | No               | No               |
| `hypothesis`          | Edge-case discovery | No            | Yes         | Yes              | No               |
| `pytest-lazy-fixture` | Fixture in params   | No            | Yes         | No               | Yes              |
| `pytest-subtests`     | Loop testing        | No            | No          | No               | No               |

---

## Real-World Pattern: **Hybrid Approach**

```python
# cases_auth.py
@case(tags="valid")
def case_valid_login(): return "admin", "secret", True

@case(tags="invalid")
def case_invalid_login(): return "guest", "wrong", False

# test_auth.py
from pytest_cases import parametrize_with_cases
import yaml

@parametrize_with_cases("u,p,ok", cases="cases_auth", has_tag="valid")
def test_valid_login(u, p, ok): ...

@param_files("data/invalid_logins.yaml")
def test_invalid_from_file(u, p, ok): ...
```

---

## Recommended Stack

| Goal               | Plugins                                 |
| ------------------ | --------------------------------------- |
| **Clean code**     | `pytest-cases`                          |
| **External data**  | `pytest-param-files` + `pytest-datadir` |
| **Edge cases**     | `hypothesis`                            |
| **Fixture params** | `pytest-lazy-fixture`                   |
| **Loop tests**     | `pytest-subtests`                       |

---

## Pro Tip: Filter Parametrized Tests

```bash
# Run only fast cases
pytest -m "fast"

# Run specific IDs
pytest -k "add_positive or login"
```

Works with `pytest-cases` tags and `ids`.

---

## Summary

| Plugin                | One-Liner                                    |
| --------------------- | -------------------------------------------- |
| `pytest-cases`        | **DRY, composable, tagged test cases**       |
| `pytest-param-files`  | **Parametrize from YAML/CSV**                |
| `hypothesis`          | **Find bugs with 1000s of generated inputs** |
| `pytest-lazy-fixture` | **Use fixtures inside `@parametrize`**       |

> **Use built-in for simple cases. Use plugins for scale, clarity, and robustness.**

Master these — and your test suite becomes **industrial-grade**.

# **pytest vs unittest** — A Comprehensive Comparison

| Feature                | **pytest**                               | **unittest** (Python Standard Library)    |
| ---------------------- | ---------------------------------------- | ----------------------------------------- |
| **Origin**             | Third-party (pip install)                | Built-in (`import unittest`)              |
| **Installation**       | `pip install pytest`                     | Already available                         |
| **Boilerplate**        | Minimal                                  | High (classes, methods, `self.assert*`)   |
| **Test Discovery**     | Automatic (`test_*.py`, `Test*` classes) | Manual or via `unittest discover`         |
| **Assertion Style**    | `assert expr` (with rich diff)           | `self.assertEqual(a, b)`, etc.            |
| **Fixtures**           | Powerful, scoped, reusable               | `setUp`/`tearDown`, `setUpClass`, limited |
| **Parametrization**    | Built-in `@pytest.mark.parametrize`      | Manual loops or `subTest` (Python 3.4+)   |
| **Plugins**            | 1000+ (cov, mock, django, etc.)          | Limited (3rd-party rare)                  |
| **Output / Reporting** | Colorful, detailed, concise              | Verbose, less readable                    |
| **Parallel Execution** | `pytest-xdist`                           | No native support                         |
| **Learning Curve**     | Gentle (Pythonic)                        | Steeper (OOP-heavy)                       |

---

## 1. **Basic Test Example**

### **unittest**

```python
# test_unittest.py
import unittest

def add(a, b):
    return a + b

class TestAdd(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)

    def test_negative(self):
        self.assertEqual(add(-1, 1), 0)

if __name__ == '__main__':
    unittest.main()
```

### **pytest**

```python
# test_pytest.py
def add(a, b):
    return a + b

def test_positive():
    assert add(2, 3) == 5

def test_negative():
    assert add(-1, 1) == 0
```

> **Winner**: **pytest** — no class, no `self`, no boilerplate.

---

## 2. **Assertions**

### **unittest**

```python
self.assertEqual(result, expected)
self.assertTrue(condition)
self.assertRaises(ValueError, func, arg)
```

### **pytest**

```python
assert result == expected           # auto-diff on failure
assert condition is True
with pytest.raises(ValueError):
    func(arg)
```

**Failure Example (pytest):**

```text
>       assert add(2, 3) == 6
E       assert 5 == 6
E        -5
E        +6
```

> **Winner**: **pytest** — richer, readable failures.

---

## 3. **Parametrized Tests**

### **unittest** (with `subTest`)

```python
class TestAdd(unittest.TestCase):
    def test_add(self):
        for a, b, exp in [(1,1,2), (2,3,5), (-1,1,0)]:
            with self.subTest(a=a, b=b):
                self.assertEqual(add(a, b), exp)
```

### **pytest**

```python
@pytest.mark.parametrize("a,b,exp", [(1,1,2), (2,3,5), (-1,1,0)])
def test_add(a, b, exp):
    assert add(a, b) == exp
```

> **Winner**: **pytest** — cleaner, scalable, better reporting.

---

## 4. **Fixtures (Setup/Teardown)**

### **unittest**

```python
class TestDB(unittest.TestCase):
    def setUp(self):
        self.db = connect_db()

    def tearDown(self):
        self.db.close()

    def test_query(self):
        result = self.db.query("SELECT 1")
        self.assertEqual(result, 1)
```

### **pytest**

```python
@pytest.fixture
def db():
    db = connect_db()
    yield db
    db.close()

def test_query(db):
    assert db.query("SELECT 1") == 1
```

**Scopes in pytest:**

```python
@pytest.fixture(scope="module")   # once per file
@pytest.fixture(scope="session")  # once per run
```

> **Winner**: **pytest** — reusable, composable, scoped.

---

## 5. **Test Discovery & Running**

### **unittest**

```bash
python -m unittest discover
# or
python test_unittest.py
```

### **pytest**

```bash
pytest                    # auto-discovers
pytest -q                 # quiet
pytest test_*.py::TestClass::test_method
pytest -k "add and not slow"   # keyword filter
```

> **Winner**: **pytest** — smarter, faster, more flexible.

---

## 6. **Mocking**

### **unittest**

```python
from unittest.mock import patch

@patch('module.func')
def test_something(self, mock_func):
    mock_func.return_value = 42
    ...
```

### **pytest**

```python
def test_something(mocker):
    mocker.patch('module.func', return_value=42)
    ...
```

> Same power — **tie**, but `mocker` fixture is cleaner.

---

## 7. **Plugins & Ecosystem**

| Plugin                    | pytest | unittest                |
| ------------------------- | ------ | ----------------------- |
| `pytest-cov`              | Yes    | `coverage.py` manual    |
| `pytest-django`           | Yes    | `DjangoTestSuiteRunner` |
| `pytest-xdist` (parallel) | Yes    | No                      |
| `pytest-benchmark`        | Yes    | No                      |
| `hypothesis`              | Yes    | Limited                 |

> **Winner**: **pytest** — unmatched ecosystem.

---

## 8. **When to Use Which?**

| Use **unittest** if:                                           | Use **pytest** if:                                      |
| -------------------------------------------------------------- | ------------------------------------------------------- |
| You **cannot install** external packages (e.g. restricted env) | You want **productivity**                               |
| Legacy codebase already uses it                                | You need **parametrization**, **fixtures**, **plugins** |
| Teaching OOP concepts                                          | You want **readable failures**                          |
| Minimal dependencies                                           | You're starting a **new project**                       |

---

## 9. **Migration: unittest → pytest**

pytest can **run unittest tests** automatically:

```bash
pytest test_unittest.py   # Just works!
```

Use `pytest --ff` to run failed tests first.

---

## 10. **Real-World Adoption**

| Framework    | Used By                                                 |
| ------------ | ------------------------------------------------------- |
| **pytest**   | Django, Flask, FastAPI, Pandas, NumPy, Dropbox, Mozilla |
| **unittest** | Legacy projects, some stdlib                            |

> **Industry standard = pytest**

---

## Summary Table

| Criteria        | **pytest** | **unittest** |
| --------------- | ---------- | ------------ |
| Syntax          | 5/5        | 2/5          |
| Assertions      | 5/5        | 3/5          |
| Parametrization | 5/5        | 2/5          |
| Fixtures        | 5/5        | 2/5          |
| Plugins         | 5/5        | 1/5          |
| Reporting       | 5/5        | 3/5          |
| Performance     | 5/5        | 4/5          |
| Learning        | 5/5        | 3/5          |

---

## Final Verdict

> **Use `pytest` for new projects.**  
> **Stick with `unittest` only if you can’t install packages.**

```bash
pip install pytest
```

Write this:

```python
def test_add():
    assert add(2, 3) == 5
```

Not this:

```python
self.assertEqual(add(2, 3), 5)
```

**pytest = The modern, Pythonic way to test.**

Here's a **line-by-line, detailed breakdown** of this **pytest assertion failure message**, with **context, meaning, and best practices**.

---

### The Test Code (Assumed)

```python
def add(a, b):
    return a + b  # ← This function is correct!

def test_add():
    assert add(2, 3) == 6  # ← This is the failing assertion
```

---

### The Full Output (What You See in Terminal)

```text
> assert add(2, 3) == 6
E assert 5 == 6
E -5
E +6
```

Let’s **dissect every line**.

---

## Line-by-Line Explanation

### `> assert add(2, 3) == 6`

- **Meaning**: This is the **exact line of code** in your test that failed.
- The `>` arrow points to the **failing assertion**.
- `add(2, 3)` was **evaluated**, and the result was compared to `6`.
- **Key Insight**: `add(2, 3)` returned `5`, not `6`.

---

### `E assert 5 == 6`

- **`E`** = **Error** (in this case, an `AssertionError`)
- **`assert 5 == 6`** = **What pytest rewrote** the assertion into.
  - pytest **evaluates both sides** of `==` and shows the **actual values**.
  - Left side: `add(2, 3)` → `5`
  - Right side: `6`
- This is **not** your original code — it's **pytest’s intelligent rewrite**.
- **Why?** To help you **debug instantly** without rerunning.

---

### `E -5` and `E +6`

These are **unified diff-style hints** showing the difference:

| Line   | Meaning                                                |
| ------ | ------------------------------------------------------ |
| `E -5` | **Expected** value was `5` (the actual result)         |
| `E +6` | **Desired** value was `6` (what you wrote in the test) |

> **Note**: This can be confusing at first — the `-` and `+` are **from the perspective of the expected value**, **not** the test.

#### Correct Interpretation:

```text
Actual result:   5   ← what the function returned
Expected result: 6   ← what you asserted
```

So:

- `-5` → "the **actual** value was 5"
- `+6` → "you **wanted** 6"

---

## Why This Happens (Even If `add` Is Correct)

Wait — `add(2, 3)` **should** return `5`, so why assert `== 6`?

### Two Possible Scenarios:

| Scenario           | Explanation                                                                          |
| ------------------ | ------------------------------------------------------------------------------------ |
| **1. Bug in Test** | You **mistyped the expected value**. The function is correct, but the test is wrong. |
| **2. Bug in Code** | The `add` function is **broken** (e.g. returns `a + b + 1`).                         |

In this case:

```python
def add(a, b):
    return a + b  # → returns 5
```

→ The **test is wrong**. It should say:

```python
assert add(2, 3) == 5
```

---

## How pytest Makes This So Helpful

| Feature                             | Benefit                  |
| ----------------------------------- | ------------------------ |
| Evaluates both sides                | No need to add `print()` |
| Shows actual vs expected            | Instant debug            |
| Unified diff (`-`/`+`)              | Visual clarity           |
| No `self.assertEqual()` boilerplate | Cleaner code             |

---

## Advanced: What If Values Are Complex?

```python
assert add([1, 2], [3]) == [1, 2, 3, 4]
```

**Output:**

```text
> assert add([1, 2], [3]) == [1, 2, 3, 4]
E assert [1, 2, 3] == [1, 2, 3, 4]
E At index 3 diff: 4 != None
E Right contains one more item: 4
E Full diff:
E - [1, 2, 3, 4]
E + [1, 2, 3]
```

→ **Deep comparison** with **detailed diff**.

---

## Best Practices

1. **Trust the output** — it’s almost always correct.
2. **Don’t add `print()`** for debugging assertions.
3. **Use `assert` freely** — pytest makes failures informative.
4. **Fix the test if the function is correct** (common in TDD).

---

## Summary: What This Output Tells You

| Part                      | Meaning                   |
| ------------------------- | ------------------------- |
| `> assert add(2, 3) == 6` | This line failed          |
| `E assert 5 == 6`         | Actual = 5, Expected = 6  |
| `E -5`                    | The function returned `5` |
| `E +6`                    | You expected `6`          |

**Conclusion**:  

> **Your test is wrong.**  
> Change `== 6` → `== 5`

---

## Pro Tip: Run with `-vv` for Even More Detail

```bash
pytest -vv
```

Shows:

- Test ID
- File and line
- Full expression tree

---

### Final Corrected Test

```python
def test_add():
    assert add(2, 3) == 5   # ← Fixed!
```

Now:

```text
test_add.py::test_add PASSED
```

**pytest’s assertion introspection is one of its killer features.**  
This output isn’t just an error — it’s a **debugging superpower**.

# **pytest Parametrized Test Failures — In-Depth Guide**

When a **parametrized test fails**, `pytest` shows **exactly which parameter set** caused the failure — with **full context**, **rich diffs**, and **isolation** from other cases.

Let’s break down **how failures work**, **what you see**, and **how to debug** effectively.

---

## 1. **Basic Parametrized Test Failure**

### Test Code

```python
# test_math.py
import pytest

@pytest.mark.parametrize("a, b, expected", [
    (1, 1, 2),
    (2, 3, 5),
    (4, 4, 9),   # ← BUG: 4+4=8, not 9
    (0, 0, 0),
])
def test_add(a, b, expected):
    assert a + b == expected
```

### Run: `pytest -q`

```text
...F.
```

### Full Failure Output

```text
=================================== FAILURES ===================================
________________________ test_add[4, 4, 9 - 4-4-expected] ________________________

a = 4, b = 4, expected = 9

    @pytest.mark.parametrize("a, b, expected", [
        (1, 1, 2),
        (2, 3, 5),
        (4, 4, 9),
        (0, 0, 0),
    ])
    def test_add(a, b, expected):
>       assert a + b == expected
E       assert 8 == 9
E        -8
E        +9

test_math.py:9: AssertionError
```

---

## Line-by-Line Breakdown

| Line                               | Meaning                                            |
| ---------------------------------- | -------------------------------------------------- |
| `test_add[4, 4, 9 - 4-4-expected]` | **Test ID** — shows **which parameter set failed** |
| `a = 4, b = 4, expected = 9`       | **Parameter values** injected into this run        |
| `> assert a + b == expected`       | **Failing line**                                   |
| `E assert 8 == 9`                  | **Actual vs Expected**                             |
| `E -8` / `E +9`                    | **Diff** — actual was `8`, expected `9`            |

> **Only one case fails** — others pass: `...F.`

---

## 2. **Custom `id` Makes Failures Readable**

### With `id=`

```python
@pytest.mark.parametrize("a, b, expected", [
    pytest.param(1, 1, 2, id="1+1"),
    pytest.param(2, 3, 5, id="2+3"),
    pytest.param(4, 4, 9, id="4+4"),   # ← fails
], ids=["one", "two", "four"])  # overrides param id
```

### Failure:

```text
____________________________ test_add[four] ____________________________

> assert a + b == expected
E assert 8 == 9
E -8
E +9
```

> **Use `id=` or `ids=`** — critical for large test matrices.

---

## 3. **Multiple Failures — All Shown**

```python
@pytest.mark.parametrize("x", [1, 2, 3, 4])
def test_even(x):
    assert x % 2 == 0
```

### Output:

```text
test_math.py: FFFF

=================================== FAILURES ===================================
____________________________ test_even[1] ____________________________
> assert 1 % 2 == 0 → 1 == 0 → False

____________________________ test_even[3] ____________________________
> assert 3 % 2 == 0 → 1 == 0 → False
```

> **All failing cases reported** — no "stop on first failure".

---

## 4. **Complex Data: Deep Diffs**

```python
@pytest.mark.parametrize("input, expected", [
    ([1, 2], [1, 2]),
    ([3, 4], [3, 5]),  # ← wrong
    ({"a": 1}, {"a": 1, "b": 2}),
])
def test_transform(input, expected):
    assert transform(input) == expected
```

### Failure:

```text
> assert {'a': 1} == {'a': 1, 'b': 2}
E Right contains one more key: 'b'
E Full diff:
E - {'a': 1, 'b': 2}
E + {'a': 1}
```

> **Deep structural comparison** — lists, dicts, sets, nested objects.

---

## 5. **Exception in Parametrized Test**

```python
@pytest.mark.parametrize("value", ["hello", 42, "world"])
def test_upper(value):
    return value.upper()
```

### Failure (on `42`):

```text
___________________________ test_upper[42] ____________________________

value = 42

    def test_upper(value):
>       return value.upper()
E       AttributeError: 'int' object has no attribute 'upper'
```

> **Exception traceback** includes **parameter values**.

---

## 6. **Using `ids` as Callable**

```python
def make_id(val):
    return f"user:{val['name']}"

@pytest.mark.parametrize("user", [
    {"name": "alice", "role": "admin"},
    {"name": "bob", "role": "guest"},
], ids=make_id)
def test_role(user):
    assert user["role"] in ["admin", "guest"]
```

### Failure:

```text
_______________________ test_role[user:bob] _______________________
> assert 'guest' in [...]
```

---

## 7. **Cartesian Product Failures**

```python
@pytest.mark.parametrize("size", ["S", "M"])
@pytest.mark.parametrize("color", ["red", "blue"])
def test_shirt(color, size):
    assert len(color + size) == 3  # ← fails for "S"+"red"
```

### Failures:

```text
test_shirt[red-S] → len("redS") = 4 ≠ 3
test_shirt[red-M] → len("redM") = 4 ≠ 3
test_shirt[blue-S] → len("blueS") = 5 ≠ 3
test_shirt[blue-M] → len("blueM") = 6 ≠ 3
```

> **All 4 combinations tested** — 4 failures.

---

## 8. **Indirect Parametrization Failure**

```python
@pytest.fixture
def setup(request):
    db = request.param
    if db == "broken":
        raise ValueError("DB failed")
    return db

@pytest.mark.parametrize("setup", ["sqlite", "broken"], indirect=True)
def test_db(setup):
    assert setup is not None
```

### Failure:

```text
________________________ test_db[broken] _________________________

> setup = request.param → "broken"
> raise ValueError("DB failed")
E ValueError: DB failed
```

> **Fixture failure** shows up in the **parametrized test ID**.

---

## 9. **Debugging Tips**

| Tip                       | Command                             |
| ------------------------- | ----------------------------------- |
| **Run only failed case**  | `pytest -k "4+4"`                   |
| **Verbose**               | `pytest -v`                         |
| **Show local variables**  | `pytest --showlocals`               |
| **Stop on first failure** | `pytest -x`                         |
| **Print in test**         | `print("debug")` → shown on failure |

---

## 10. **Best Practices**

| Do                         | Avoid                      |
| -------------------------- | -------------------------- |
| Use `id=` for clarity      | Rely on `[0]`, `[1]`       |
| Keep parametrization small | 100+ cases → hard to debug |
| Use `ids=` callable        | Hardcoded strings          |
| Group related cases        | Mix unrelated logic        |

---

## Summary: What You See on Failure

| Element                    | Purpose                |
| -------------------------- | ---------------------- |
| `test_func[param1-param2]` | **Which case failed**  |
| `a = 1, b = 2`             | **Injected values**    |
| `assert 3 == 4`            | **Actual vs expected** |
| `-3` / `+4`                | **Diff**               |
| Full traceback             | **Where** it failed    |

---

## Final Example: Perfect Failure Report

```python
@pytest.mark.parametrize("a,b,expected", [
    pytest.param(1, 1, 2, id="1+1"),
    pytest.param(2, 3, 6, id="2+3"),  # ← bug
], ids=lambda x: x.upper())
```

**Failure:**

```text
________________________ test_add[2+3] ________________________

a = 2, b = 3, expected = 6

>   assert a + b == expected
E   assert 5 == 6
E    -5
E    +6
```

**Fix:**

```python
pytest.param(2, 3, 5, id="2+3")
```

---

**pytest parametrized failures are not just errors — they are precise, actionable diagnostics.**  
Master the output → **debug 10x faster**.

# **pytest Mocking Techniques — Complete Guide**

**Mocking** = replacing real objects with **controlled fakes** to **isolate** and **test** behavior.

`pytest` + `unittest.mock` (or `pytest-mock`) = **powerful, clean, Pythonic mocking**.

---

## 1. **Core Tools**

| Tool            | Purpose                         |
| --------------- | ------------------------------- |
| `unittest.mock` | Built-in (Python 3.3+)          |
| `pytest-mock`   | Thin wrapper (`mocker` fixture) |
| `MagicMock`     | Default mock object             |
| `patch`         | Replace object in namespace     |

---

## 2. **Installation**

```bash
pip install pytest pytest-mock
```

> `pytest-mock` is **optional** but **highly recommended**.

---

## 3. **Basic Mocking with `mocker` (pytest-mock)**

### Why `mocker`?

- Auto-undo patches
- Cleaner syntax
- No manual `stop()`

### Example: Mock a Function

```python
# app.py
def get_user(id):
    return api_call(f"/users/{id}")

def api_call(url):
    import requests
    return requests.get(url).json()
```

```python
# test_app.py
def test_get_user(mocker):
    # Mock the API call
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"name": "Alice"}

    mocker.patch("app.requests.get", return_value=mock_response)

    from app import get_user
    user = get_user(123)

    assert user["name"] == "Alice"
    # Verify it was called
    app.requests.get.assert_called_once_with("/users/123")
```

---

## 4. **Key Mocking Patterns**

### 1. **Mock Return Value**

```python
mocker.patch("module.func", return_value=42)
assert module.func() == 42
```

### 2. **Mock Side Effect**

```python
def fake_div(x):
    if x == 0:
        raise ValueError("div by zero")
    return 100 / x

mocker.patch("calc.divide", side_effect=fake_div)
```

### 3. **Mock Raises Exception**

```python
mocker.patch("db.connect", side_effect=ConnectionError("DB down"))
```

### 4. **Mock Object Attributes**

```python
mock_db = mocker.Mock()
mock_db.execute.return_value = [("Alice",)]
mocker.patch("app.db", mock_db)
```

---

## 5. **Where to Patch? — The Golden Rule**

> **Patch where the object is *used*, not where it’s defined.**

```python
# WRONG
mocker.patch("database.connect")  # if used as db.connect

# CORRECT
mocker.patch("app.db.connect")    # if app.py does: from database import connect as db
```

---

## 6. **Mocking Classes & Methods**

```python
# user.py
class User:
    def save(self):
        return db.save(self)

# test_user.py
def test_user_save(mocker):
    mock_db = mocker.Mock()
    mock_db.save.return_value = True
    mocker.patch("user.db", mock_db)

    u = User()
    assert u.save() is True
    mock_db.save.assert_called_once_with(u)
```

---

## 7. **Mocking with `spec` — Prevent Typos**

```python
mock = mocker.Mock(spec=User)  # Only allows real methods
mock.save()        # OK
mock.nonexistent() # AttributeError
```

---

## 8. **Advanced: `wraps` — Keep Real Behavior**

```python
real_div = lambda x, y: x / y

mock_div = mocker.patch("calc.div", wraps=real_div)
mock_div.side_effect = lambda x, y: 0 if y == 0 else real_div(x, y)
```

---

## 9. **Mocking Context Managers**

```python
mock_file = mocker.Mock()
mock_file.__enter__.return_value = "fake content"
mock_file.__exit__.return_value = None

mocker.patch("builtins.open", return_value=mock_file)

with open("file.txt") as f:
    assert f == "fake content"
```

---

## 10. **Spying — Monitor Real Calls**

```python
def test_spy(mocker):
    real_func = app.process_data
    spy = mocker.spy(app, "process_data")

    app.main()
    spy.assert_called_once_with({"x": 1})
```

---

## 11. **Mocking Built-ins**

```python
mocker.patch("builtins.print")
app.log("hello")
print.assert_called_with("hello")
```

---

## 12. **Mocking External APIs (requests)**

```python
def test_api(mocker):
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": "ok"}

    mocker.patch("requests.get", return_value=mock_resp)

    result = fetch_data()
    assert result == "ok"
```

---

## 13. **Mocking with Fixtures**

```python
@pytest.fixture
def mock_db(mocker):
    db = mocker.Mock()
    mocker.patch("app.db", db)
    return db

def test_with_fixture(mock_db):
    mock_db.get.return_value = "data"
    assert app.get_data() == "data"
```

---

## 14. **Common Pitfalls**

| Pitfall                   | Fix                                    |
| ------------------------- | -------------------------------------- |
| Patching wrong module     | Use `app.module`, not `module`         |
| Forgetting `return_value` | Mock returns `None` → `NoneType` error |
| Not using `assert_called` | Miss side-effect bugs                  |
| Over-mocking              | Mock only what’s needed                |

---

## 15. **Best Practices**

| Do                       | Avoid                     |
| ------------------------ | ------------------------- |
| Use `mocker` fixture     | Manual `patch.start/stop` |
| Use `spec=`              | Allow fake attributes     |
| Use `assert_called_with` | Assume it was called      |
| Keep mocks close to test | Global mocks              |
| Name mocks clearly       | `mock`, `mock1`           |

---

## 16. **Real-World Example: API Client**

```python
# client.py
import requests

def get_user(id):
    resp = requests.get(f"https://api.example.com/users/{id}")
    resp.raise_for_status()
    return resp.json()["name"]

# test_client.py
def test_get_user_success(mocker):
    mock_resp = mocker.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"name": "Alice"}
    mock_resp.raise_for_status.return_value = None

    mocker.patch("requests.get", return_value=mock_resp)

    assert get_user(1) == "Alice"

def test_get_user_404(mocker):
    mock_resp = mocker.Mock()
    mock_resp.status_code = 404
    mock_resp.raise_for_status.side_effect = requests.HTTPError()

    mocker.patch("requests.get", return_value=mock_resp)

    with pytest.raises(requests.HTTPError):
        get_user(999)
```

---

## 17. **Debugging Mocks**

```bash
pytest -s                    # see print() in mocks
pytest --showlocals         # see mock values
mocker.patch(..., new_callable=mocker.spy)
```

---

## Summary Table

| Technique            | Use Case                  |
| -------------------- | ------------------------- |
| `return_value`       | Fake return               |
| `side_effect`        | Dynamic behavior / errors |
| `spec=`              | Type safety               |
| `wraps=`             | Partial mock              |
| `mocker.spy`         | Monitor real calls        |
| `assert_called_with` | Verify input              |

---

## Final Tip: **Mock Only the Boundary**

```text
[Your Code] → [External Dependency]
              ↑
           Mock this
```

> **Don’t mock your own logic.**

---

## Resources

- `pytest-mock`: https://github.com/pytest-dev/pytest-mock
- `unittest.mock`: https://docs.python.org/3/library/unittest.mock.html
- Book: *"Effective Python Testing with pytest"* 

---

**Master mocking → write fast, reliable, isolated tests.**  
With `pytest-mock`, it’s **clean, safe, and fun**.
