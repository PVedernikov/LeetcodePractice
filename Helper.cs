using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LeetcodePreapare
{
    public static class Helper
    {
        public static int[] GetBits(int n)
        {
            var result = new int[32];
            for (int i = 0; i < 32; i++)
            {
                if ((n & (1 << i)) >= 0)
                {
                    result[31 - i] = 1;
                }
            }
            return result;
        }

        public static string GetBitsString(int n)
        {
            var result = new char[32];
            for (int i = 0; i < 32; i++)
            {
                result[31 - i] = ((n & (1 << i)) > 0) ? '1' : '0';
            }

            return new string(result);
        }

        public static string GetBitsString(uint n)
        {
            var result = new char[32];
            for (int i = 0; i < 32; i++)
            {
                result[31 - i] = ((n & (1u << i)) > 0) ? '1' : '0';
            }

            return new string(result);
        }
    }
}
