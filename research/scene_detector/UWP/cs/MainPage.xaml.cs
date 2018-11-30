using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.AI.MachineLearning;
using Windows.Storage;
using Windows.Media;
using Windows.Graphics.Imaging;
using System.Threading.Tasks;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.Storage.Pickers;
using Windows.UI.Xaml.Media.Imaging;
using System.Diagnostics;
using Newtonsoft.Json;
using Windows.Media.Playback;
using Windows.Media.Core;
using Microsoft.Graphics.Canvas;

namespace RacingViewsDetection
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private const string                _kModelFileName = "model-new.onnx";
        private const string                _kLabelsFileName = "Labels.json";
        private LearningModel               _model = null;
        private LearningModelSession        _session;
        private List<string>                _labels = new List<string>();
        private int                         _runCount = 0;
        private MediaPlayer mediaPlayer;

        public MainPage()
        {
            this.InitializeComponent();
        }

        void Reset()
        {
            // let everything go in reverse order, taking care to dispose
            _session.Dispose();
            _session = null;
            _model.Dispose();
            _model = null;
        }

        LearningModelDeviceKind GetDeviceKind()
        {
            switch (Combo_DeviceKind.SelectedIndex)
            {
                case 0:
                    return LearningModelDeviceKind.Default;
                case 1:
                    return LearningModelDeviceKind.Cpu;
                case 2:
                    return LearningModelDeviceKind.DirectX;
                case 3:
                    return LearningModelDeviceKind.DirectXHighPerformance;
                case 4:
                    return LearningModelDeviceKind.DirectXMinPower;
            }
            return LearningModelDeviceKind.Default;
        }

        /// <summary>
        /// Load the label and model files
        /// </summary>
        /// <returns></returns>
        private async Task LoadModelAsync()
        {
            // just load the model one time.
            if (_model != null) return;

            StatusBlock.Text = $"Loading {_kModelFileName} ... patience ";

            try
            {
                // Parse labels from label json file.  We know the file's 
                // entries are already sorted in order.
                var fileString = File.ReadAllText($"Assets/{_kLabelsFileName}");
                var fileDict = JsonConvert.DeserializeObject<Dictionary<string,string>>(fileString);
                foreach( var kvp in fileDict)
                {
                    _labels.Add(kvp.Value);
                }

                // Load and create the model 
                var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_kModelFileName}"));
                _model = await LearningModel.LoadFromStorageFileAsync(modelFile);

                // Create the evaluation session with the model and device
                _session = new LearningModelSession(_model, new LearningModelDevice(GetDeviceKind()));
            }
            catch (Exception ex)
            {
                StatusBlock.Text = $"error: {ex.Message}";
                _model = null;
            }
        }

        private async void ButtonReset_Click(object sender, RoutedEventArgs e)
        {
            Reset();

            // set the button states
            ButtonLoad.IsEnabled = true;
            ButtonRun.IsEnabled = false;
            ButtonReset.IsEnabled = false;

            StatusBlock.Text = "Model unloaded";
        }

        private async void ButtonLoad_Click(object sender, RoutedEventArgs e)
        {
            int ticks = Environment.TickCount;

            // Load the model
            await LoadModelAsync();

            ticks = Environment.TickCount - ticks;

            // set the button states
            ButtonLoad.IsEnabled = false;
            ButtonRun.IsEnabled = true;
            ButtonReset.IsEnabled = true;

            StatusBlock.Text = $"Model loaded in { ticks } ticks, ready to run";
        }

        /// <summary>
        /// Trigger file picker and image evaluation
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void ButtonRun_Click(object sender, RoutedEventArgs e)
        {
            ButtonRun.IsEnabled = false;
            UIPreviewImage.Source = null;
            StatusBlock.Text = "Loading image...";
            try
            {
                // Trigger file picker to select an image file
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.FileTypeFilter.Add(".mp4");

                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                StorageFile selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();

                if (selectedStorageFile.FileType.ToLower() == ".mp4") {

                    UIMediaPlayerElement.Source = MediaSource.CreateFromStorageFile(selectedStorageFile);
                    UIMediaPlayerElement.TransportControls.ShowAndHideAutomatically = false;
                    mediaPlayer = UIMediaPlayerElement.MediaPlayer;
                    
                    //mediaPlayer.Source = MediaSource.CreateFromStorageFile(selectedStorageFile); //.CreateFromUri(new Uri("ms-appx:///Assets/example_video.mkv"));
                    mediaPlayer.VideoFrameAvailable += MediaPlayer_VideoFrameAvailable;
                    mediaPlayer.IsVideoFrameServerEnabled = true;
                    //mediaPlayer.Play();
                    
                }

                if (selectedStorageFile.FileType.ToLower() == ".jpg" || selectedStorageFile.FileType.ToLower() == ".png")
                {

                    SoftwareBitmap softwareBitmap;
                    using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                    {
                        // Create the decoder from the stream 
                        BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                        // Get the SoftwareBitmap representation of the file in BGRA8 format
                        softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                        softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);

                    }

                    // Display the image
                    SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                    await imageSource.SetBitmapAsync(softwareBitmap);
                    UIPreviewImage.Source = imageSource;

                    // Encapsulate the image within a VideoFrame to be bound and evaluated
                    VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                    await EvaluateVideoFrameAsync(inputImage);
                }
            }
            catch (Exception ex)
            {
                StatusBlock.Text = $"error: {ex.Message}";
                ButtonRun.IsEnabled = true;
            }
        }

        private async void MediaPlayer_VideoFrameAvailable(MediaPlayer sender, object args)
        {
            CanvasDevice canvasDevice = CanvasDevice.GetSharedDevice();
            await Dispatcher.RunAsync(Windows.UI.Core.CoreDispatcherPriority.Normal, async () =>
            {
                SoftwareBitmap softwareBitmapImg;
                SoftwareBitmap frameServerDest = new SoftwareBitmap(BitmapPixelFormat.Bgra8, 500, 500, BitmapAlphaMode.Premultiplied);

                using (CanvasBitmap canvasBitmap = CanvasBitmap.CreateFromSoftwareBitmap(canvasDevice, frameServerDest))
                {
                    sender.CopyFrameToVideoSurface(canvasBitmap);
                    softwareBitmapImg = await SoftwareBitmap.CreateCopyFromSurfaceAsync(canvasBitmap, BitmapAlphaMode.Ignore);
                }

                SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                await imageSource.SetBitmapAsync(softwareBitmapImg);
                UIPreviewImage.Source = imageSource;

                // Encapsulate the image within a VideoFrame to be bound and evaluated
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmapImg);

                await EvaluateVideoFrameAsync(inputImage);
            });
        }


        /// <summary>
        /// Evaluate the VideoFrame passed in as arg
        /// </summary>
        /// <param name="inputFrame"></param>
        /// <returns></returns>
        private async Task EvaluateVideoFrameAsync(VideoFrame inputFrame)
        {
            if (inputFrame != null)
            {
                try
                {
                    //StatusBlock.Text = "Binding image...";

                    // create a binding object from the session
                    LearningModelBinding binding = new LearningModelBinding(_session);

                    // bind the input image
                    ImageFeatureValue imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputFrame);
                    binding.Bind("data", imageTensor);

                    //StatusBlock.Text = "Running model...";

                    int ticks = Environment.TickCount;

                    // Process the frame with the model
                    var results = await _session.EvaluateAsync(binding, $"Run { ++_runCount } ");

                    ticks = Environment.TickCount - ticks;

                    // retrieve results from evaluation
                    var resultTensor = results.Outputs["classLabel"] as TensorString;
                    var resultVector = resultTensor.GetAsVectorView();

                    var resultLoss = results.Outputs["loss"] as IList<Dictionary<string, float>>;

                    // Find the top 3 probabilities
                    List<float> topProbabilities = new List<float>() { 0.0f, 0.0f, 0.0f };
                    List<int> topProbabilityLabelIndexes = new List<int>() { 0, 0, 0 };
                    //// SqueezeNet returns a list of 1000 options, with probabilities for each, loop through all
                    //for (int i = 0; i < resultVector.Count(); i++)
                    //{
                    //    // is it one of the top 3?
                    //    for (int j = 0; j < 3; j++)
                    //    {
                    //        if (resultVector[i] > topProbabilities[j])
                    //        {
                    //            topProbabilityLabelIndexes[j] = i;
                    //            topProbabilities[j] = resultVector[i];
                    //            break;
                    //        }
                    //    }
                    //}

                    //// Display the result
                    //string message = $"Run took { ticks } ticks";
                    //for (int i = 0; i < 3; i++)
                    //{
                    //    message += $"\n\"{ _labels[topProbabilityLabelIndexes[i]]}\" with confidence of { topProbabilities[i]}";
                    //}
                    StatusBlock.Text = resultVector[0];
                }
                catch (Exception ex)
                {
                    StatusBlock.Text = $"error: {ex.Message}";
                }

                ButtonRun.IsEnabled = true;
            }
        }
    }
}
