class ClassA {
    int a = 10;
    int funcAdd(int x, int y) {
        return x + y + a;
    }
}

public class ex01 {
    public static void main(String[] args) {
        int x = 3, y = 6, r;
        ClassA cal = new ClassA();
        r = cal.funcAdd(x, y);
        System.out.println(r);
    }
}