"""
Setup Script für TradingBot
Installiert das Package und alle Dependencies
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Lese README für long_description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Lese requirements.txt
requirements_path = this_directory / "requirements.txt"
with open(requirements_path) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Version aus __init__.py oder default
VERSION = "1.0.0"

setup(
    name="tradingbot-alpaca",
    version=VERSION,
    description="Professioneller TradingBot für Alpaca Markets mit ML-Strategien",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TradingBot Team",
    author_email="contact@tradingbot.com",
    url="https://github.com/your-username/TradingBot",
    
    # Packages
    packages=find_packages(),
    include_package_data=True,
    
    # Dependencies
    install_requires=requirements,
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Entry points for CLI
    entry_points={
        'console_scripts': [
            'tradingbot=main:main',
            'tradingbot-validate=run_validation:main',
            'tradingbot-dashboard=dashboard.app:main',
        ],
    },
    
    # Package data
    package_data={
        '': ['*.json', '*.txt', '*.md', '*.html', '*.css', '*.js'],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Keywords
    keywords="trading bot algorithmic trading machine learning alpaca markets finance",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-username/TradingBot/issues",
        "Source": "https://github.com/your-username/TradingBot",
        "Documentation": "https://github.com/your-username/TradingBot#readme",
    },
    
    # Development dependencies
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ]
    },
)