# Contributing to AI Trading Assistant

Thank you for your interest in contributing to the AI Trading Assistant project!

## Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/ai_trading_assistant.git
   cd ai_trading_assistant
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add docstrings to functions
   - Update documentation if needed

3. **Test your changes**
   ```bash
   streamlit run src/main.py
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes

## Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable names

### Documentation
- Add docstrings to all functions
- Update README.md for major changes
- Add examples for new features

### Example Function Documentation
```python
def calculate_metric(data: pd.Series, period: int = 14) -> float:
    """
    Calculate specific metric for given data
    
    Args:
        data: Input data series
        period: Calculation period
    
    Returns:
        Calculated metric value
    
    Example:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> result = calculate_metric(data, period=5)
    """
    pass
```

## Areas for Contribution

### High Priority
- [ ] Additional ML models (XGBoost, Random Forest)
- [ ] More technical indicators
- [ ] Enhanced error handling
- [ ] Unit tests
- [ ] Performance optimizations

### Medium Priority
- [ ] Crypto currency support
- [ ] Real-time data streaming
- [ ] Advanced charting options
- [ ] Export functionality
- [ ] User authentication

### Low Priority
- [ ] Dark/light theme toggle
- [ ] Mobile responsiveness
- [ ] Multi-language support
- [ ] Advanced filters

## Bug Reports

When reporting bugs, include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- System information (OS, Python version)
- Error messages and logs

## Feature Requests

For feature requests, include:
- Clear description of the feature
- Use case and benefits
- Proposed implementation (optional)
- Examples from similar applications

## Testing

Currently, testing is done manually. Future contributions should include:
- Unit tests for new functions
- Integration tests for modules
- UI tests for Streamlit components

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Help with setup
- Discussion of new features
- General feedback

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards others

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- README.md Contributors section
- Release notes
- Project documentation

Thank you for contributing! 🎉
