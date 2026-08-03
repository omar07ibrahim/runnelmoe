# Independent oracle

The M1 Python/PyTorch oracle will implement the documented tiny-model equations
with a deliberately different module and control-flow structure from the Rust
runtime. It produces routes, logits, and token fixtures; it is not imported by
production code.
