#include<thread>
#include<iostream>
#include<string>
#include<future>
#include<ctype.h>
using namespace std;
string Func(int a, const string& b){
    printf("num is %d",a);
    tolower(b);
    return b;
}

int main(){
    string b = "HEllo";
    // thread 初始化一个线程
    thread t1(Func,250,b);
    t1.join();
    // async 来自于头文件future,初始化一个异步线程
    auto fut = std::async(Func,250,b);
    cout<<fut.get()<<endl;
    return 0;

}