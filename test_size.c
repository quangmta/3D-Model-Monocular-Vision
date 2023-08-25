#include <stdio.h>
# pragma pack (16)
struct test_size
{
    char a;
    int b;
};
int main()
{
    printf("size = %d",sizeof(struct test_size));
    return 0;
}
