# Makefile for LaTeX project

.PHONY: build clean view help

build:
	mkdir -p build
	latexmk -pdf -output-directory=build main.tex

clean:
	latexmk -C -output-directory=build

view: build
	@command -v xdg-open > /dev/null && xdg-open build/main.pdf || open build/main.pdf

help:
	@echo "Available targets:"
	@echo "  make build  - Build the PDF document"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make view   - Build and open the PDF"
	@echo "  make help   - Show this help message"
