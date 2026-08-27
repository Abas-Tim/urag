class MainWindow
{
    private readonly BoolToVisibilityConverter _converter = new BoolToVisibilityConverter();
    public static object OpenCommand { get; set; }

    public MainWindow()
    {
        Initialize();
    }

    public void Show()
    {
        Render();
    }

    private void Render()
    {
        OnLoaded();
    }

    private void OnLoaded()
    {
    }

    private void Initialize()
    {
    }
}

class Item
{
    public string Name { get; set; }
}
