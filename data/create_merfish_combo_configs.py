import json
import pathlib


def main():
    merfish_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/ABC_handoff/expression_matrices/MERFISH-C57BL6J-638850/20230630')

    out_dir = merfish_dir.parent.parent
    out_dir = out_dir / (merfish_dir.parent.name+'-combined')
    out_dir = out_dir / '20230630'
    #assert not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dir = pathlib.Path('../configs')
    assert merfish_dir.is_dir()
    assert config_dir.is_dir()

    log_path_list = [
        n for n in merfish_dir.iterdir()
        if n.is_file() and 'log' in n.name]

    log_path_list.sort()

    raw_path_list = [
        n for n in merfish_dir.iterdir()
        if n.is_file() and 'raw' in n.name]

    raw_path_list.sort()

    assert len(log_path_list) == len(raw_path_list)
    all_paths = [n for n in merfish_dir.iterdir()]
    assert len(raw_path_list)+len(log_path_list) == len(all_paths)

    root = 'C57BL6J-638850'
    log_config = {
        'h5ad_path_list': [str(p.resolve().absolute()) for p in log_path_list],
        'output_path': str((out_dir / f'{root}.log2.h5ad').resolve().absolute())}

    raw_config = {
        'h5ad_path_list': [str(p.resolve().absolute()) for p in raw_path_list],
        'output_path': str((out_dir / f'{root}.raw.h5ad').resolve().absolute())}

    with open(config_dir / 'merfish_log_combo.json', 'w') as out_file:
        out_file.write(json.dumps(log_config))
    with open(config_dir / 'merfish_raw_combo.json', 'w') as out_file:
        out_file.write(json.dumps(raw_config))

if __name__ == "__main__":
    main()
