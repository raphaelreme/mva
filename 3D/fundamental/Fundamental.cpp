// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse
// Date:     2013/10/08

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};


// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}

// Create the matrix A from matches.
// m number of matches to consider.
Matrix<float> generateA(vector<Match> matches, int m) {
    if (m < 8) {
        cerr << "Not enough data to create the matrix A" << endl;
        exit(1);
    }
    // In the case n = 8. A is augmented of one zero line in order to be squarred.
    Matrix<float> A(max(9, m), 9);

    A.fill(0);
    for (int i=0; i<m; i++) {
        Match match = matches[i];
        FloatPoint3 P1, P2;
        P1[0] = 0.001*match.x1; P1[1] = 0.001*match.y1; P1[2] = 1;
        P2[0] = 0.001*match.x2; P2[1] = 0.001*match.y2; P2[2] = 1;

        /* xi^t.F.x'i = 0 <=> Ai^t.F = 0
        * With Ai = (x[0]*x'[0], x[0]*x'[1], x[0]*x'[2], x[1]*x'[0], x[1]*x'[1],....)
        */
        for (int k=0; k<3; k++){
            for (int l=0; l<3; l++) {
                A(i, 3*k + l) = P1[k] * P2[l];
            }
        }
    }

    return A;
}

// Compute F such as AF ~= 0 and rank of F is 2.
FMatrix<float,3,3> computeFfromA(Matrix<float>& A) {
    // Trying to minimize |AF| = F^t.A^t.A.F with |F| = 1 (scale is indifferent)
    // F is an eignenvector of AT.A associated with the smallest eigenvalue
    // -> in the SVD decomposition of A, F is V_9
    Vector<float> S1(min(A.nrow(), A.ncol())); // = 9 in practive
    Matrix<float> U1(A.nrow(),A.nrow()), V1(A.ncol(),A.ncol());
    svd(A, U1, S1, V1);

    // To ensure rank of F = 2, let's unflatten F and set the smallest eigenvalue of F at 0
    // using another SVD decomposition.
    FMatrix<float,3,3> F;
    for (int i=0; i<3; i++){
        for (int j=0; j<3; j++) {
            F(i, j) = V1(8, 3*i+j);
        }
    }
    FVector<float,3> S2;
    FMatrix<float,3,3> U2, V2;
    svd(F, U2, S2, V2);
    S2[2] = 0;

    return U2 * Diagonal(S2) * V2;
}

// RANSAC algorithm to compute F from point matches (8-point algorithm)
// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    int Niter=100000; // Adjusted dynamically
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    FMatrix<float,3,3> F;
    vector<int> inliers;

    vector<Match> matches_shuffled = matches;

    // Normalization matrix: P' = NP
    FMatrix<float,3,3> N;
    N.fill(0);
    N(0,0) = 0.001;
    N(1,1) = 0.001;
    N(2,2) = 1;

    // Random generator
    random_device rd;
    mt19937 g(rd());

    int iteration = 0;
    while (iteration < Niter) {
        iteration++;

        // Randomly shuffling matches and use the 8 first matches to create A.
        shuffle(matches_shuffled.begin(), matches_shuffled.end(), g);
        Matrix<float> A = generateA(matches_shuffled, 8);
        F = N*computeFfromA(A)*N;

        // Compute inliers indexes with this particular F.
        // xi^t.F.xi' = 0 <=> (F^t.xi)^t.xi' = 0 <=> x_i' belongs to the line defined by the vector F^t.xi.
        // Inliers will be the matches such that the distance of xi' to the line F^t.xi is smaller than distMax.
        inliers.clear();
        for (int i=0; i<matches.size(); i++) {
            Match match = matches[i];
            FloatPoint3 P1, P2;
            P1[0] = match.x1; P1[1] = match.y1; P1[2] = 1;
            P2[0] = match.x2; P2[1] = match.y2; P2[2] = 1;

            FVector<float,3> line = transpose(F) * P1;
            float dist_squarred = pow(line*P2, 2) / (line[0]*line[0] + line[1]*line[1]);

            if (dist_squarred < distMax * distMax){
                inliers.push_back(i);
            }
        }

        // Update best F/inliers and therefore Niter.
        if (inliers.size() >  bestInliers.size()) {
            bestF = F;
            bestInliers = inliers;
            float m = bestInliers.size();
            float n = matches.size();

            cout << "Update best at iteration " << iteration << endl;
            cout << "Nb of inliers: " << m << endl;

            // If m/n if smaller than 0.1, (m/n)**8 is too close to 0 and therefore 1/log(1 - (m/n)**8) explodes.
            // Only update the number of iterations when at least 10% of the data are considered as inliers.
            if (m/n > 0.1) {
                Niter = log(BETA)/log(1-pow(m/n, 8));
            }
        }
    }

    cout << "End of ransac: " << iteration << " iterations!" << endl;

    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    // Recompute F on all inliers !
    // It doesn't seems to improve at all the results... Especially for the grass.
    // Comment those two lines in order to keep the bestF found with the ransac algorithm.
    Matrix<float> A = generateA(matches, matches.size());
    return N*computeFfromA(A)*N;

    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2,
                     const FMatrix<float,3,3>& F) {
    int w = I1.width();
    int h = I1.height();
    FloatPoint3 P;

    while (true) {
        int x,y;
        if(getMouse(x,y) == 3)
            break;

        drawCircle(x, y, 3, Color(255, 0, 0), 1);

        if (x<w) { // Click on image 1!
            P[0] = x; P[1] = y; P[2] = 1;
            // Line = F^t.xi
            FVector<float,3> line = transpose(F) * P;

            // line[0]*x + line[1]*y + line[2] = 0
            if (abs(line[0]) < abs(line[1])) {  // y = -line[0]/line[1]*x - line[2]/line[1]
                drawLine(w, -1*line[2]/line[1], 2*w, -1*w*line[0]/line[1] - line[2]/line[1], Color(255, 0, 0));
            } else {  // x = -line[1]/line[0]*y - line[2]/line[0]
                drawLine(w - line[2]/line[0], 0, w - h*line[1]/line[0] - line[2]/line[0], h, Color(255, 0, 0));
            }
        } else {  // Click on image 2!
            P[0] = x-w; P[1] = y; P[2] = 1;
            // Line = F.xi'
            FVector<float,3> line = F * P;

            // line[0]*x + line[1]*y + line[2] = 0
            if (abs(line[0]) < abs(line[1])) {  // y = -line[0]/line[1]*x - line[2]/line[1]
                drawLine(0, -1*line[2]/line[1], w, -1*w*line[0]/line[1] - line[2]/line[1], Color(255, 0, 0));
            } else {  // x = -line[1]/line[0]*y - line[2]/line[0]
                drawLine(0 - line[2]/line[0], 0, 0 - h*line[1]/line[0] - line[2]/line[0], h, Color(255, 0, 0));
            }
        }
    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    cout << " matches: " << matches.size() << endl;
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}
