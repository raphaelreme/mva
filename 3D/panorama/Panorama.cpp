// Imagine++ project
// Project:  Panorama
// Author:   Pascal Monasse
// Date:     2013/10/08

#include <Imagine/Graphics.h>
#include <Imagine/Images.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <sstream>
using namespace Imagine;
using namespace std;

// Record clicks in two images, until right button click
void getClicks(Window w1, Window w2,
               vector<IntPoint2>& pts1, vector<IntPoint2>& pts2) {
    // Completed code
    IntPoint2 position;
    Window clicked_window;
    int sub;

    for (uint i=0;; i++) {
        // The more correspondances we have the better it gets.
        int click = anyGetMouse(position, clicked_window, sub);
        if (click == 3) {
            return;  // Finish points acquisition on a right click.
        }
        if (click == 2) {
            continue;  // Ignore middle clicks.
        }

        /* In case of left click:
         * - Add the point to the vectors
         * - Display it on the figure in order to help acquisition of future points.
         */
        if (clicked_window == w1) {
            pts1.push_back(position);
            setActiveWindow(w1);
            drawCircle(position, 2, Color(255, 0, 0), 3);
            drawString(position, to_string(pts1.size()), Color(255, 0, 0), 15);
        } else if (clicked_window == w2) {
            pts2.push_back(position);
            setActiveWindow(w2);
            drawCircle(position, 2, Color(255, 0, 0), 3);
            drawString(position, to_string(pts2.size()), Color(255, 0, 0), 15);
        }
    }
}

// Return homography compatible with point matches
Matrix<float> getHomography(const vector<IntPoint2>& pts1,
                            const vector<IntPoint2>& pts2) {
    size_t n = min(pts1.size(), pts2.size());
    if(n<4) {
        cout << "Not enough correspondences: " << n << endl;
        return Matrix<float>::Identity(3);
    }
    Matrix<double> A(2*n,8);
    Vector<double> B(2*n);

    // Completed code
    int x, y, x_p, y_p;
    A.fill(0);
    for (size_t i=0; i<n; i++) {
        x = pts1[i].x(); y = pts1[i].y();
        x_p = pts2[i].x(); y_p = pts2[i].y();

        // First equation: A_{2i} . h = B_{2i}
        A(2*i, 0) = x; A(2*i, 1) = y; A(2*i, 2) = 1;
        A(2*i, 6) = -x*x_p; A(2*i, 7) = -y*x_p;
        B[2*i] = x_p;

        // Second equation A_{2i+1} . h = B_{2i+1}
        A(2*i+1, 3) = x; A(2*i+1, 4) = y; A(2*i+1, 5) = 1;
        A(2*i+1, 6) = -x*y_p; A(2*i+1, 7) = -y*y_p;
        B[2*i+1] = y_p;
    }
    // Can be resolve in one system: A . h = B
    // End
    B = linSolve(A, B);
    Matrix<float> H(3, 3);
    H(0,0)=B[0]; H(0,1)=B[1]; H(0,2)=B[2];
    H(1,0)=B[3]; H(1,1)=B[4]; H(1,2)=B[5];
    H(2,0)=B[6]; H(2,1)=B[7]; H(2,2)=1;

    // Sanity check
    for(size_t i=0; i<n; i++) {
        float v1[]={(float)pts1[i].x(), (float)pts1[i].y(), 1.0f};
        float v2[]={(float)pts2[i].x(), (float)pts2[i].y(), 1.0f};
        Vector<float> x1(v1,3);
        Vector<float> x2(v2,3);
        x1 = H*x1;
        cout << x1[1]*x2[2]-x1[2]*x2[1] << ' '
             << x1[2]*x2[0]-x1[0]*x2[2] << ' '
             << x1[0]*x2[1]-x1[1]*x2[0] << endl;
    }
    return H;
}

// Grow rectangle of corners (x0,y0) and (x1,y1) to include (x,y)
void growTo(float& x0, float& y0, float& x1, float& y1, float x, float y) {
    if(x<x0) x0=x;
    if(x>x1) x1=x;
    if(y<y0) y0=y;
    if(y>y1) y1=y;    
}

// Panorama construction
void panorama(const Image<Color,2>& I1, const Image<Color,2>& I2,
              Matrix<float> H) {
    Vector<float> v(3);
    float x0=0, y0=0, x1=I2.width(), y1=I2.height();

    v[0]=0; v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=0; v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=I1.width(); v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    v[0]=0; v[1]=I1.height(); v[2]=1;
    v=H*v; v/=v[2];
    growTo(x0, y0, x1, y1, v[0], v[1]);

    cout << "x0 x1 y0 y1=" << x0 << ' ' << x1 << ' ' << y0 << ' ' << y1<<endl;

    Image<Color> I(int(x1-x0), int(y1-y0));
    setActiveWindow( openWindow(I.width(), I.height()) );
    I.fill(WHITE);

    // Completed partie
    Matrix<float> H_inv = inverse(H);
    const Color *color_1, *color_2;
    for (int x=0; x<I.width(); x++) {
        for (int y=0; y<I.height(); y++) {
            color_1 = nullptr; color_2 = nullptr;
            v[0] = x + x0; v[1] = y + y0; v[2] = 1;
            if (int(v[0]) >= 0 && int(v[0]) < I2.width() && int(v[1]) >= 0 && int(v[1]) < I2.height()) {
                color_2 = &I2(int(v[0]), int(v[1]));
            }

            v = H_inv * v;
            v /= v[2];
            if (int(v[0]) >= 0 && int(v[0]) < I1.width() && int(v[1]) >= 0 && int(v[1]) < I1.height()) {
                color_1 = &I1(int(v[0]), int(v[1]));
            }

            if (color_1) {
                if (color_2) {
                    // In case images overlaps, let's take the mean.
                    // Seems smoother that take only the pixels from a particular image.
                    // But the overlapping resulting region is therefore blurred.
                    int r, g, b;
                    r = int((float(color_1->r()) + float(color_2->r()))/2);
                    g = int((float(color_1->g()) + float(color_2->g()))/2);
                    b = int((float(color_1->b()) + float(color_2->b()))/2);
                    I(x, y) = Color(byte(r), byte(g), byte(b));
                } else {
                    I(x, y) = *color_1;
                }
            } else if (color_2) {
                I(x, y) = *color_2;
            }
        }
    }
    // End of completed party.
    display(I,0,0);
}

// Main function
int main(int argc, char* argv[]) {
    const char* s1 = argc>1? argv[1]: srcPath("image0006.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("image0007.jpg");

    // Load and display images
    Image<Color> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load the images" << endl;
        return 1;
    }
    Window w1 = openWindow(I1.width(), I1.height(), s1);
    display(I1,0,0);
    Window w2 = openWindow(I2.width(), I2.height(), s2);
    setActiveWindow(w2);
    display(I2,0,0);

    // Get user's clicks in images
    vector<IntPoint2> pts1, pts2;
    getClicks(w1, w2, pts1, pts2);

    vector<IntPoint2>::const_iterator it;
    cout << "pts1="<<endl;
    for(it=pts1.begin(); it != pts1.end(); it++)
        cout << *it << endl;
    cout << "pts2="<<endl;
    for(it=pts2.begin(); it != pts2.end(); it++)
        cout << *it << endl;

    // Compute homography
    Matrix<float> H = getHomography(pts1, pts2);
    cout << "H=" << H/H(2,2);

    // Apply homography
    panorama(I1, I2, H);

    endGraphics();
    return 0;
}
