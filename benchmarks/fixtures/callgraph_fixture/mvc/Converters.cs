class BoolToVisibilityConverter
{
    public object Convert(object value)
    {
        return value;
    }
}

class LegacyWidget
{
    public void Draw()
    {
        Paint();
    }

    private void Paint()
    {
    }
}

class LegacyTools
{
    public static int Parse(string input)
    {
        return input.Length;
    }
}
