#!/usr/bin/env python3
"""
Setup script for Media Generation Example using uv.
Handles virtual environment creation and environment configuration.
"""
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def check_python_version() -> None:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10+ is required")
        sys.exit(1)
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"✓ Python {version} detected")


def check_uv_installed() -> None:
    """Check if uv is installed."""
    try:
        run_command(["uv", "--version"])
        print("✓ uv is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ uv is not installed")
        print("Please install uv first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  # or with pip: pip install uv")
        sys.exit(1)


def create_virtual_environment() -> None:
    """Create virtual environment using uv."""
    venv_path = Path(".venv")

    if venv_path.exists():
        print("✓ Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == "y":
            shutil.rmtree(venv_path)
        else:
            return

    print("Creating virtual environment with uv...")
    run_command(["uv", "venv"])
    print("✓ Virtual environment created")


def install_dependencies() -> None:
    """Install dependencies using uv."""
    print("Installing dependencies with uv...")
    run_command(["uv", "sync"])
    print("✓ Dependencies installed")


def install_dev_dependencies() -> None:
    """Install development dependencies."""
    print("Installing development dependencies...")
    run_command(["uv", "sync", "--extra", "dev"])
    print("✓ Development dependencies installed")


def setup_environment_file() -> None:
    """Set up the .env file from template."""
    env_file = Path(".env")
    example_file = Path("env.example")

    if env_file.exists():
        print("✓ .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != "y":
            return

    if example_file.exists():
        shutil.copy(example_file, env_file)
        print("✓ Created .env file from template")
    else:
        print("✗ env.example file not found")
        return


def main() -> None:
    """Main setup function."""
    print("Media Generation Example Setup (uv)")
    print("=" * 40)

    # Check Python version
    check_python_version()

    # Check uv installation
    check_uv_installed()

    # Create virtual environment
    create_virtual_environment()

    # Install dependencies
    install_dependencies()

    # Install dev dependencies
    install_dev_dependencies()

    # Setup environment file
    setup_environment_file()

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit the .env file with your actual API keys")
    print("2. Activate the virtual environment:")
    print("   source .venv/bin/activate")
    print("3. Or use uv to run commands directly:")
    print("   uv run python example_usage.py")
    print("4. Run tests:")
    print("   uv run pytest")
    print("\nRequired API keys:")
    print("- OPENAI_API_KEY: For DALL-E image generation")
    print("- REPLICATE_API_TOKEN: For various AI models")
    print("\nUseful uv commands:")
    print("- uv run <command>: Run a command in the virtual environment")
    print("- uv add <package>: Add a new dependency")
    print("- uv add --dev <package>: Add a development dependency")
    print("- uv sync: Install/update dependencies")
    print("- uv lock: Update lock file")


if __name__ == "__main__":
    main()
