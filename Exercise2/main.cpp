#include <iostream>
#include <Eigen/Dense>
#include <cmath>
using namespace std;
using namespace Eigen;

//Fattorizzazione PA=LU
void palu(const MatrixXd& A, const VectorXd& b, const VectorXd& x)
{
    VectorXd x_tilde=A.lu().solve(b);
    cout<<"Soluzione col metodo PA=LU: "<<endl;
    cout<<x_tilde<<endl;
    double err_rel=(x_tilde-x).norm()/x.norm();
    cout<<"Errore relativo col metodo PA=LU: "<<endl;
    cout<<err_rel<<endl;
}

//Fattorizzazione QR
void qr(const MatrixXd& A, const VectorXd& b, const VectorXd& x)
{
    VectorXd x_tilde=A.householderQr().solve(b);
    cout<<"Soluzione col metodo QR: "<<endl;
    cout<<x_tilde<<endl;
    double err_rel=(x_tilde-x).norm()/x.norm();
    cout<<"Errore relativo col metodo QR: "<<endl;
    cout<<err_rel<<endl;
}

int main()
{
    Vector2d x;
    Vector2d b1,b2,b3;
    Matrix2d A1,A2,A3;
    
    x<<-1.0e+0,-1.0e+00;

    A1<<5.547001962252291e-01,-3.770900990025203e-02,8.320502943378437e-01,-9.992887623566787e-0;
    b1<<-5.169911863249772e-01, 1.672384680188350e-01;

    A2<<5.547001962252291e-01, -5.540607316466765e-01,8.320502943378437e-01, -8.324762492991313e-01;
    b2<<-6.394645785530173e-04, 4.259549612877223e-04;

    A3<<5.547001962252291e-01, -5.547001955851905e-01,8.320502943378437e-01, -8.320502947645361e-01; 
    b3<<-6.400391328043042e-10, 4.266924591433963e-10;

    cout<<"Sistema 1:"<<endl;
    palu(A1,b1,x);
    qr(A1,b1,x);

    cout<<"Sistema 2:"<<endl;
    palu(A2,b2,x);
    qr(A2,b2,x);

    cout<<"Sistema 3:"<<endl;
    palu(A3,b3,x);
    qr(A3,b3,x);

    return 0;
}
