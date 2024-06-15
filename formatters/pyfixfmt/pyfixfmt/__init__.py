import glob
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import Union

import autoflake
import autotyping.autotyping
import black
import isort
from isort.exceptions import FileSkipComment
from libcst.codemod import CodemodContext
from libcst.codemod import exec_transform_with_prettyprint

from pyfixfmt.config import Config
from pyfixfmt.config import resolve_config


def main() -> None:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--force", action="store_true")
    arg_parser.add_argument("--config")
    arg_parser.add_argument("--file-glob")
    arg_parser.add_argument("--no-import-flaking", action="store_true")
    arg_parser.add_argument("--verbose", action="store_true")

    args = arg_parser.parse_args()

    explicit_config: str = args.config
    file_glob: str = args.file_glob
    no_import_flaking: bool = args.no_import_flaking
    is_verbose: bool = args.verbose
    is_forced: bool = args.force

    if is_verbose:
        print("File Glob: ", file_glob)

    files_to_evaluate = tuple(glob.glob(file_glob, recursive=True))
    format_files(files_to_evaluate, explicit_config, is_verbose, no_import_flaking, is_forced)


def format_files(
    files: Iterable[Union[str, Path]],
    pyproject_path: Optional[Union[str, Path]] = None,
    is_verbose: bool = False,
    no_import_flaking: bool = False,
    is_forced: bool = False,
) -> None:
    files = tuple(files)
    source_root = black.find_project_root(files)[0]

    config = resolve_config(source_root, explicit_path=pyproject_path, is_verbose=is_verbose)

    if no_import_flaking:
        if is_verbose:
            print("Disabling import flaking")
        config.disable_import_flaking()

    for file in files:
        file_path = Path(file)
        if is_verbose:
            print("Formatting: ", file_path)
        run_all_fixers_on_path(file_path, config, is_forced)


def run_all_fixers_on_path(file_path: Path, config: Config, is_forced) -> None:
    with file_path.open("r") as file_reader:
        original_source = file_reader.read()

    # if there is a drafty comment, skip this file
    if not is_forced:
        if any(x.lstrip().startswith("# GEN") for x in original_source.splitlines()):
            print("Skipping file with drafty comment: ", file_path)
            return

    source = run_all_fixers_on_str(file_path, original_source, config)

    if original_source != source:
        with file_path.open("w") as file_writer:
            file_writer.write(source)


def run_all_fixers_on_str(file_path: Optional[Path], source: str, config: Config) -> str:
    source = run_autotyping(file_path, source, config)
    source = run_autoflake(file_path, source, config)
    source = run_isort(file_path, source, config)
    source = run_black(file_path, source, config)

    return source


def run_autotyping(path: Optional[Path], file_source: str, config: Config) -> str:
    if _do_any_files_match(path, config.do_not_autotype):
        return file_source

    command_instance = autotyping.autotyping.AutotypeCommand(
        CodemodContext(), none_return=True, scalar_return=True, annotate_magics=True
    )
    fixed_source: str = exec_transform_with_prettyprint(command_instance, file_source)
    if fixed_source is None or (not len(fixed_source) and len(file_source.strip())):
        raise Exception(f'autotyping was unable to parse {path or "the source"}')

    return fixed_source


def run_autoflake(path: Optional[Path], file_source: str, config: Config, remove_unused_imports: bool = True) -> str:
    # Just skip some files completely.
    if _do_any_files_match(path, config.do_not_do_anything_with_imports):
        return file_source

    # For safe keeping, we're going to make sure that we don't remove unused imports from unexpected places
    if _do_any_files_match(path, config.do_not_remove_imports):
        remove_unused_imports = False

    fixed_source: str = autoflake.fix_code(
        file_source,
        additional_imports=None,
        expand_star_imports=False,
        remove_all_unused_imports=remove_unused_imports,
        ignore_init_module_imports=False,
    )

    return fixed_source


def run_isort(path: Optional[Path], file_source: str, config: Config) -> str:
    # Just skip some files completely.
    if _do_any_files_match(path, config.do_not_do_anything_with_imports):
        return file_source

    try:
        output = isort.code(file_source, config=config.isort_config)
        return output
    except FileSkipComment:
        return file_source


def run_black(_path: Optional[Path], file_source: str, config: Config) -> str:
    mode = config.black_config

    try:
        return black.format_file_contents(src_contents=file_source, fast=True, mode=mode)
    except black.NothingChanged:
        return file_source


def _do_any_files_match(path: Optional[Path], files: Iterable[str]) -> bool:
    if path is None:
        return False

    absolute_path = str(path.absolute())
    return any(do_not_remove_file in absolute_path for do_not_remove_file in files)
