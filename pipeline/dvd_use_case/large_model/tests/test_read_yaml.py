from pathlib import Path
from utilities.read_yaml import read_yaml

def test_read_yaml(tmp_path):
    yaml_content = '''
    foo: bar
    num: 42
    nested:
        a: 1
        b: 2
    '''
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    result = read_yaml(yaml_file)
    assert result['foo'] == 'bar'
    assert result['num'] == 42
    assert result['nested']['a'] == 1
    assert result['nested']['b'] == 2
