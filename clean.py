import os
import sys
import shutil
import zipfile

def clear_directory(directory_path):
    """
    Clear all contents of the specified directory.
    """
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def truncate_log_file(log_file_path):
    """
    Truncate the specified log file.
    """
    try:
        with open(log_file_path, 'w'):
            pass
        print(f"The log file {log_file_path} has been truncated.")
    except Exception as e:
        print(f"Failed to truncate the log file {log_file_path}. Reason: {e}")

def backup_directory(directory_path, backup_path):
    """
    Backup the specified directory to a zip file.
    """
    try:
        shutil.make_archive(backup_path, 'zip', directory_path)
        print(f"Backup of {directory_path} created at {backup_path}.zip")
    except Exception as e:
        print(f"Failed to backup {directory_path}. Reason: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean.py <directory1> <directory2> ... [--truncate-log <log_file>] [--backup <backup_directory>]")
        sys.exit(1)

    truncate_log = False
    log_file = None
    backup_directories = []

    if '--truncate-log' in sys.argv:
        truncate_log = True
        log_file_index = sys.argv.index('--truncate-log') + 1
        if log_file_index < len(sys.argv):
            log_file = sys.argv[log_file_index]
        else:
            print("Please specify the log file to truncate after --truncate-log.")
            sys.exit(1)

    if '--backup' in sys.argv:
        backup_index = sys.argv.index('--backup') + 1
        if backup_index < len(sys.argv):
            backup_directories = sys.argv[backup_index:]
        else:
            print("Please specify the directories to backup after --backup.")
            sys.exit(1)

    directories = [arg for arg in sys.argv[1:] if arg not in ['--truncate-log', '--backup'] and arg != log_file]

    for directory in directories:
        if backup_directories:
            backup_directory(directory, os.path.join("backup", os.path.basename(directory)))
        clear_directory(directory)

    if truncate_log and log_file:
        truncate_log_file(log_file)
