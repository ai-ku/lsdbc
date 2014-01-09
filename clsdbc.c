/**************************************************************************
** CLSDBC: C implementation for Locally Scaled Density Based Clustering
** 
** Version: 1.0 Date: 19/07/2006
** Version: 1.1 Date: 21/02/2014
** 
** Author: Ergun Bicici
** 
** Copyright: Ergun Bicici, 19/07/2006, 21/02/2014.
** 
** Citation: 
** Ergun Biçici and Deniz Yuret. Locally Scaled Density Based Clustering. 
** In Proceedings of the 8th International Conference on Adaptive and Natural Computing Algorithms (ICANNGA 2007), LNCS 4431, volume 4431, Warsaw, Poland, pages 739--748, April 2007.
** 
** Link:
** http://home.ku.edu.tr/~ebicici/publications/Year/2007.html
** 
** Available from:
** https://github.com/ai-ku/lsdbc
**
** Bibtex entry:
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
}
** 
***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h> /* For INT_MAX */
#include <string.h>
#include <float.h> /* For DBL_MAX */

#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
#define SAFE_FREE(a) {if (a) {free(a); a = NULL;}}

/* Distance matrix format. */
typedef struct dmat *DMat;

/* Row-major dense matrix.  Rows are consecutive vectors. */
struct dmat {
  unsigned numPoints;
  double **value; /* Accessed by [row][col]. Free value[0] and value to free.*/
};

typedef struct {
    double val;  /* Eps */
    unsigned pointNum;
} Node;

typedef struct {
    double Eps;
    unsigned pointNum;
    int class;
    unsigned *neighbors;
} Point;


/* Function Declarations. */
int compare (const void * a, const void * b);
void LSDBC(DMat DistMatrix, unsigned k, int numDimension, double alpha, FILE * ofp);
void ExpandCluster(unsigned size, unsigned p, int ClusterID, unsigned k, double powerValue);
void kNNDistVal(double *v, unsigned size, unsigned p, unsigned k, double *Eps, unsigned *neighbors);
int localMax(unsigned pointNum, unsigned k);
void printUsage(char *progname);
DMat NewDMat(unsigned numPoints);
void FreeDMat(DMat D);
void FreePoints(Point * points);
static DMat LoadDenseMatrix(FILE *file);

/*End of Function Declarations*/

/* Global Variables */
Point * Points;
int clusterNoise = 0;

int main(int argc, char *argv[]) {

   FILE *ofp, *ifp;
   char *ifname, *ofname;
   double alpha;
   int numDimension;
   unsigned k;

   DMat DistMatrix = NULL;

   if (!((argc == 2) || (argc == 6))) {
       printUsage(argv[0]);
       exit(1); 
   }
   
   ifname = (char *)malloc(200*sizeof(char));
   strcpy(ifname,argv[1]);
   if (argc == 6) {
     k = atoi(argv[2]);
     alpha = atof(argv[3]);
     numDimension = atoi(argv[4]);
     clusterNoise = atoi(argv[5]);
   }
   else {
     printf("Using default parameters:");
     k = 7;
     alpha = 3;
     numDimension = 2;
     clusterNoise = 0;
   }
   printf("\tk:%d, alpha:%f, n:%d, clusterNoise:%d\n", (int) k, (double) alpha, (int) numDimension, (int) clusterNoise);
   
   ifp = fopen(ifname,"r");
   if (ifp == 0) { fprintf(stderr,"File %s not found\n",ifname); exit(1); }
   
   ofname = (char *)malloc(200*sizeof(char));
   strcpy(ofname,ifname);
   strcat(ofname, ".out");
   ofp = fopen(ofname, "w");
   if (ofp == NULL) {fprintf(stderr, "Can't open output file %s!\n", ofname); exit(1);}
   
   DistMatrix = LoadDenseMatrix(ifp);
   if (!DistMatrix) {fprintf(stderr,"Failed to read dense matrix."); exit(1);}
   else {fprintf(stderr,"Distance matrix %s is read.\n", ifname);}
   
   if (k > (int) DistMatrix->numPoints) {
     fprintf(stderr,"k: %d is greater than the number of points: %d.", k, (int)DistMatrix->numPoints);
     fprintf(stderr,"You may want to check the format of %s\n", ifname);
     exit(1);
   }
   
   /* [clusts,noise] = LSDBC(D, k, n, alpha) */
   LSDBC(DistMatrix, k, numDimension, alpha, ofp);
   
   FreeDMat(DistMatrix);
   FreePoints(Points);
   
   fclose(ifp);
   fclose(ofp);
   fprintf(stderr,"Output clustering is written to %s.\n", ofname);
   return 0;

}

/* Row major order.  Rows are vectors that are consecutive in memory.  
   Matrix is initialized to empty. */
DMat NewDMat(unsigned numPoints) {
  unsigned i;
  DMat D = (DMat) malloc(sizeof(struct dmat));
  if (!D) {errmesg("NewDMat"); return NULL;}
  D->numPoints = numPoints;

  D->value = (double **) malloc(numPoints * sizeof(double *));
  if (!D->value) {SAFE_FREE(D); return NULL;}

  D->value[0] = (double *) calloc(numPoints * numPoints, sizeof(double));
  if (!D->value[0]) {SAFE_FREE(D->value); SAFE_FREE(D); return NULL;}

  for (i = 1; i < numPoints; i++) 
      D->value[i] = D->value[i-1] + numPoints;

  return D;
}

void FreeDMat(DMat D) {
  if (!D) return;
  SAFE_FREE(D->value[0]);
  SAFE_FREE(D->value);
  free(D);
}

void FreePoints(Point * points) {
  if (!points) return;
  SAFE_FREE(points->neighbors);
  SAFE_FREE(points);
}

/* Static functions are only available in its own file. */
DMat LoadDenseMatrix(FILE *file) {
  unsigned numPoints, i, j;
  DMat D;
  if (fscanf(file, " %ld", &numPoints) != 1) {
    errmesg("LoadDenseMatrix: bad file format");
    return NULL;
  }

  D = NewDMat(numPoints);
  if (!D) return NULL;

  for (i = 0; i < numPoints; i++)
    for (j = 0; j < numPoints; j++) {
      if (fscanf(file, " %lf", &(D->value[i][j])) != 1) {
        errmesg("LoadDenseMatrix: bad file format");
        return NULL;
      }
    }
  return D;
}

void printUsage(char *progname) {
  fprintf(stderr,"CLSDBC Version 1.0\n" 
	  "\twritten by Ergun Bicici.\n\n");
  fprintf(stderr,"\tUsage: %s matrix_file k alpha numDimensions \n\n", progname); 
  fprintf(stderr,"[integer] k: Number of neighbors to consider (for kNN based density estimation). \n[double] alpha: Adjusting parameter for density cutoff. \n[integer] numDimensions: Number of dimensions the original data resides in. \n[integer] clusterNoise: 0 or 1 (do not cluster noise or cluster)\n\talpha = numDimensions \t--> Cluster number is changed once the density falls below the half of the original density. \n"); 
  exit(1);
}

/* Compare function for qsort. */
int compare (const void * a, const void * b)
{
    return (int) INT_MAX * (((Node *)a)->val - ((Node *)b)->val ); 
}

/* [clusts,noise] = LSDBC(D, k, n, alpha) 

% Given a similarity matrix for a number of points
% allocates all points to a cluster or specify them as noise
% D: Distance matrix
% k: k-dist parameter
% n: number of dimensions
*/
void LSDBC(DMat DistMatrix, unsigned k, int numDimension, double alpha, FILE * ofp) {
  
    int ClusterID, NoiseClusterID, numofNoise;
    unsigned i, j, size;
    Node * EpsMatrix;
    double powerValue;
    
    size = DistMatrix->numPoints;
    
    /* Checking for memory allocation */
    if ((EpsMatrix = (Node*)calloc(size, sizeof(Node))) == NULL) {
	printf("Unable to allocate memory\n");
	exit(1);
    }

    /* Checking for memory allocation */
    if ((Points = (Point*)calloc(size, sizeof(Point))) == NULL) {
	printf("Unable to allocate memory\n");
	exit(1);
    }

    /* Initialization */
    for (i=0; i<size; i++) {

	Points[i].class = 0;
	Points[i].pointNum = i;

	/* v = DistMatrix->value[i]; */

	/* Checking for memory allocation */
	if ((Points[i].neighbors = (unsigned*)calloc(k, sizeof(unsigned))) == NULL) {
	    printf("Unable to allocate memory\n");
	    exit(1);
	}

	kNNDistVal(DistMatrix->value[i], size, i, k, &Points[i].Eps, Points[i].neighbors);

	EpsMatrix[i].pointNum = i;
	EpsMatrix[i].val = Points[i].Eps;
    }

    ClusterID = 1;
    NoiseClusterID = -1;
    
    qsort(EpsMatrix, size, sizeof(Node), compare);

    powerValue = pow(2,(double) alpha/numDimension);

    /* Main Loop */
    for (i=0; i<size; i++) {
	j = EpsMatrix[i].pointNum;
	if ((Points[j].class == 0) && (localMax(j, k))) {
	    ExpandCluster(size, j, ClusterID, k, powerValue);
	    ClusterID++;
	}
    }
    
    ClusterID--; /* ClusterID was increased additionally. */
    if (clusterNoise) {
        for (i=0; i<size; i++) {
	    j = EpsMatrix[i].pointNum;
	    if (Points[j].class == 0) {
	        ExpandCluster(size, j, NoiseClusterID, k, powerValue);
		NoiseClusterID--;
	    }
	}
        NoiseClusterID++;
        fprintf(ofp, "%d clusters, %d noise clusters\n", ClusterID, -NoiseClusterID);
    }
    else
        fprintf(ofp, "%d clusters\n", ClusterID);
    
    numofNoise = 0;
    /* % The remaining points which are unclassified become noise. */
    for (i=0; i<size; i++) {
	if (Points[i].class <= 0) {
	    if (Points[i].class == 0)
	        Points[i].class = NoiseClusterID;
	    numofNoise++;
	}
	/* Convert Points into a readable output form. */
	fprintf(ofp, "%d ", Points[i].class);
    }
    
    fprintf(ofp, "\n\n%d points are classified as noise.\n", numofNoise);
    
    SAFE_FREE(EpsMatrix);
}

/* % Decides whether the point is a local max among its neighbors */
int localMax(unsigned pointNum, unsigned k) {

    unsigned i;
    unsigned * neighbors = Points[pointNum].neighbors;

    for (i=0; i<k; i++) {
	if (Points[neighbors[i]].Eps < Points[pointNum].Eps)
	    return 0;
    }

    return 1;
}


/* % Finds the k-dist of a given point given the similarity matrix */
void kNNDistVal(double *v, unsigned size, unsigned p, unsigned k, double *Eps, unsigned *neighbors) {

    unsigned i, j, maxInd;
    double maxDist;
    Node * minKMatrix;
    Node * restMatrix;

    /* Checking for memory allocation */
    if ((minKMatrix = (Node*)calloc(k, sizeof(Node))) == NULL) {
	printf("Unable to allocate memory\n");
	exit(1);
    }

    /* Checking for memory allocation */
    if ((restMatrix = (Node*)calloc(size-k, sizeof(Node))) == NULL) {
	printf("Unable to allocate memory\n");
	exit(1);
    }

    for (i=0; i<k; i++) {
	minKMatrix[i].val = v[i];
	minKMatrix[i].pointNum = i;
    }

    for (i=k; i<size; i++) {
	restMatrix[i-k].val = v[i];
	restMatrix[i-k].pointNum = i;
    }

    /* To skip the point itself. */
    if (p < k)
	minKMatrix[p].val = DBL_MAX;
    else
	restMatrix[p-k].val = DBL_MAX;

    /* Find the maxDist among the first k points. */
    maxDist = minKMatrix[0].val;
    maxInd = 0;
    for (i=1; i<k; i++) {
	if (minKMatrix[i].val > maxDist) {
	    maxDist = minKMatrix[i].val;
	    maxInd = i;
	}
    }

    /* Go through the rest of the points. */
    for (j=0; j<size - k; j++) {
	if (restMatrix[j].val < maxDist) {
	    maxDist = minKMatrix[maxInd].val = restMatrix[j].val;
	    minKMatrix[maxInd].pointNum = restMatrix[j].pointNum;
	    
	    for (i=0; i<k; i++) {
		if (minKMatrix[i].val > maxDist) {
		    maxDist = minKMatrix[i].val;
		    maxInd = i;
		}
	    }
	}
    }

    *Eps = maxDist;
    for (i=0; i<k; i++) {
	neighbors[i] = minKMatrix[i].pointNum;
    }

}

/* % Expands the cluster of a given point */
void ExpandCluster(unsigned size, unsigned p, int ClusterID, unsigned k, double powerValue) {

    unsigned expansionIndex, pointIndex, i, numofElements, currentP;
    unsigned * seeds;
    double powVal;

    /* Checking for memory allocation */
    if ((seeds = (unsigned*)calloc(size, sizeof(unsigned))) == NULL) {
	printf("Unable to allocate memory\n");
	exit(1);
    }

    Points[p].class = ClusterID;

    /* % We remove already clustered points from consideration */
    expansionIndex = 0;
    for (i=0; i<k; i++) {
	/* % Is this check needed? Yes - this is missing in DBSCAN paper, which 
	   % introduces a bug. */
	if (Points[Points[p].neighbors[i]].class <= 0) {
	    Points[Points[p].neighbors[i]].class = ClusterID;
	    seeds[expansionIndex++] = Points[p].neighbors[i];
	}
    }

    numofElements = expansionIndex;
    pointIndex = 0;
    while (numofElements > 0) {
	currentP = seeds[pointIndex];
	if (Points[currentP].Eps <= powerValue * Points[p].Eps) {
	    for (i=0; i<k; i++) {
	      if ((ClusterID > 0 && Points[Points[currentP].neighbors[i]].class <= 0) || (ClusterID < 0 && Points[Points[currentP].neighbors[i]].class == 0)) {
		    Points[Points[currentP].neighbors[i]].class = ClusterID;
		    seeds[expansionIndex++] = Points[currentP].neighbors[i];
		    numofElements++;
		}
	    }
	}
	pointIndex++;
	numofElements--;
    }

}

