using Log = Common.Logging;

class Boot
{
    void Main()
    {
        Log.Info("hi");
        var window = new MainWindow();
        window.Show();
    }
}
