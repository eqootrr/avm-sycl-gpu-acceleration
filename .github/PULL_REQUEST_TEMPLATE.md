## Description
Brief description of changes in this PR.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Fixes # (issue number)

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] All existing tests pass

### Test Commands
```bash
# Commands to run tests
mkdir build && cd build
cmake .. -DAVM_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

## Checklist
- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated if needed
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests pass

## Performance Impact
If this PR affects performance, describe the impact:
- Benchmark results before/after:
- Memory usage changes:
- GPU kernel changes:

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any other information reviewers should know.
