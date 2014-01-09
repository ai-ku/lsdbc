LSDBC
=====

Locally Scaled Density Based Clustering

CLSDBC: C implementation for Locally Scaled Density Based Clustering

Version: 1.0 Date: 19/07/2006

Version: 1.1 Date: 21/02/2014

Author: Ergun Bicici

Copyright: Ergun Bicici, 19/07/2006, 21/02/2014.


Citation:

Ergun Biçici and Deniz Yuret. Locally Scaled Density Based Clustering. In Proceedings of the 8th International Conference on Adaptive and Natural Computing Algorithms (ICANNGA 2007), LNCS 4431, volume 4431, Warsaw, Poland, pages 739-748, April 2007.

Link:

http://home.ku.edu.tr/~ebicici/publications/Year/2007.html


This code is also available here: https://github.com/ai-ku/lsdbc


Bibtex item:

@inproceedings{Bicici:ICANNGA07,
title = "Locally Scaled Density Based Clustering",
author = "Ergun Bi{\c{c}}ici and Deniz Yuret",
booktitle = "Proceedings of the 8th International Conference on Adaptive and Natural Computing Algorithms (ICANNGA 2007), LNCS 4431",
year = "2007",
pages = "739--748",
journal = {Lecture Notes in Computer Science},
volume = {4431},
isbn = {978-3-540-71589-4},
month = "April",
address = "Warsaw, Poland",
keywords = "Machine Learning",
pdf = "http://home.ku.edu.tr/~ebicici/publications/2007/LSDBC/LSDBC-icannga07.pdf",
ps = {http://home.ku.edu.tr/~ebicici/publications/2007/LSDBC/LSDBC-icannga07.ps},
note = {Springer-Verlag} 
}


/* [clusts,noise] = LSDBC(D, k, n, alpha) 

% Given a similarity matrix for a number of points
% allocates all points to a cluster or specify them as noise
% D: Distance matrix
% k: k-dist parameter
% n: number of dimensions
*/

For compiling: 

gcc -o clsdbc clsdbc.c -lm

For debugging with kdgb: 

gcc -g -o clsdbc clsdbc.c -lm

Usage: %s [options] matrix_file

Input matrix_file format: Dense text.

Example:

3

0   0.1        4.2

0.1 0        2.2

4.2 2.2        0

