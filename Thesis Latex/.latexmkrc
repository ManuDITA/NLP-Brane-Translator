# latexmkrc for VU/UvA Master Thesis

$pdf_mode = 1;              # Use pdflatex
$postscript_mode = 0;
$dvi_mode = 0;

$out_dir = 'build';         # Output directory
$aux_dir = 'build';         # Auxiliary files directory
$pdf_previewer = 'start okular %S';
@default_files = ('main.tex');

$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Use bibtex (template uses \bibliography{}, not biblatex)
$bibtex_use = 1;
