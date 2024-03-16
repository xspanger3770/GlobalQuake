package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.earthquake.quality.QualityClass;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.time.Instant;
import java.time.format.DateTimeFormatter;

public class GraphicsSettingsPanel extends SettingsPanel {

    private JCheckBox chkBoxScheme;
    private JSlider sliderFpsIdle;
    private JCheckBox chkBoxEnableTimeFilter;
    private JTextField textFieldTimeFilter;

    private JCheckBox chkBoxEnableMagnitudeFilter;
    private JTextField textFieldMagnitudeFilter;
    private JSlider sliderOpacity;
    private JComboBox<String> comboBoxDateFormat;
    private JCheckBox chkBox24H;
    private JCheckBox chkBoxDeadStations;
    private JSlider sliderIntensityZoom;
    private JTextField textFieldMaxArchived;
    private JSlider sliderStationsSize;
    private JRadioButton[] colorButtons;

    // Cinema mode
    private JTextField textFieldTime;
    private JSlider sliderZoomMul;

    private JCheckBox chkBoxEnableOnStartup;
    private JCheckBox chkBoxReEnable;
    private JCheckBox chkBoxDisplayMagnitudeHistogram;
    private JCheckBox chkBoxDisplaySystemInfo;
    private JCheckBox chkBoxDisplayQuakeAdditionalInfo;
    private JCheckBox chkBoxAlertBox;
    private JCheckBox chkBoxTime;
    private JCheckBox chkBoxShakemap;
    private JCheckBox chkBoxCityIntensities;
    private JCheckBox chkBoxCapitals;
    private JComboBox<QualityClass> comboBoxQuality;

    private JCheckBox chkBoxClusters;
    private JCheckBox chkBoxClusterRoots;
    private JCheckBox chkBoxHideClusters;
    private JCheckBox chkBoxAntialiasStations;
    private JCheckBox chkBoxAntialiasClusters;

    private JCheckBox chkBoxAntialiasOldQuakes;

    private JCheckBox chkBoxAntialiasQuakes;


    public GraphicsSettingsPanel() {
        setLayout(new BorderLayout());

        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("General", createGeneralTab());
        tabbedPane.addTab("Old Events", createEventsTab());
        tabbedPane.addTab("Stations", createStationsTab());
        tabbedPane.addTab("Cinema Mode", createCinemaModeTab());

        add(tabbedPane, BorderLayout.CENTER);
    }

    private Component createGeneralTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel generalPanel = createVerticalPanel(false);

        //JPanel performancePanel = createVerticalPanel("Performance");
        JPanel performancePanel = createGridBagPanel("Performance");

        sliderFpsIdle = new JSlider(SwingConstants.HORIZONTAL, 10, 200, Settings.fpsIdle);
        sliderFpsIdle.setPaintLabels(true);
        sliderFpsIdle.setPaintTicks(true);
        sliderFpsIdle.setMajorTickSpacing(10);
        sliderFpsIdle.setMinorTickSpacing(5);

        JLabel label = new JLabel("FPS limit: %s".formatted(sliderFpsIdle.getValue()));
        performancePanel.add(label, createGbc(0));
        sliderFpsIdle.addChangeListener(e -> label.setText("FPS limit: %s".formatted(sliderFpsIdle.getValue())));
        performancePanel.add(sliderFpsIdle, createGbc(1));

        generalPanel.add(performancePanel);

        JPanel dateFormatPanel = createGridBagPanel("Date and Time setting");

        dateFormatPanel.add(new JLabel("Preferred date format:"), createGbc(0, 0));

        JPanel comboBoxDateFormatPanel = new JPanel();
        comboBoxDateFormat = new JComboBox<>();
        Instant now = Instant.now();
        for (DateTimeFormatter formatter : Settings.DATE_FORMATS) {
            comboBoxDateFormat.addItem(formatter.format(now));
        }
        comboBoxDateFormat.setSelectedIndex(Settings.selectedDateFormatIndex);
        comboBoxDateFormatPanel.add(comboBoxDateFormat);
        comboBoxDateFormatPanel.add(chkBox24H = new JCheckBox("Use 24 hours format.", Settings.use24HFormat));
        dateFormatPanel.add(alignLeft(comboBoxDateFormatPanel), createGbc(1, 0));

        generalPanel.add(dateFormatPanel);

        JPanel mainWindowPanel = createGridBagPanel("Main screen");
        mainWindowPanel.add(chkBoxDisplaySystemInfo = new JCheckBox("Display system info.", Settings.displaySystemInfo), createGbc(0, 0));
        mainWindowPanel.add(chkBoxDisplayMagnitudeHistogram = new JCheckBox("Display magnitude histogram.", Settings.displayMagnitudeHistogram), createGbc(1, 0));
        mainWindowPanel.add(chkBoxDisplayQuakeAdditionalInfo = new JCheckBox("Display technical earthquake data.", Settings.displayAdditionalQuakeInfo), createGbc(0, 1));
        mainWindowPanel.add(chkBoxAlertBox = new JCheckBox("Display alert box for nearby earthquakes.", Settings.displayAlertBox), createGbc(1, 1));
        mainWindowPanel.add(chkBoxShakemap = new JCheckBox("Display shakemap hexagons.", Settings.displayShakemaps), createGbc(0, 2));
        mainWindowPanel.add(chkBoxTime = new JCheckBox("Display time.", Settings.displayTime), createGbc(1, 2));
        mainWindowPanel.add(chkBoxCityIntensities = new JCheckBox("Display estimated intensities in cities.", Settings.displayCityIntensities), createGbc(0, 3));
        mainWindowPanel.add(chkBoxCapitals = new JCheckBox("Display capital cities.", Settings.displayCapitalCities), createGbc(1, 3));

        generalPanel.add(mainWindowPanel);

        JPanel clustersPanel = createGridBagPanel("Cluster settings");
        clustersPanel.add(chkBoxClusterRoots = new JCheckBox("Display Clusters (possible shaking locations).", Settings.displayClusterRoots), createGbc(0));
        clustersPanel.add(chkBoxClusters = new JCheckBox("Display Stations assigned to Clusters (local mode only).", Settings.displayClusters), createGbc(1));
        clustersPanel.add(chkBoxHideClusters = new JCheckBox("Hide Cluster after the Earthquake is actually found.", Settings.hideClustersWithQuake), createGbc(2));

        generalPanel.add(clustersPanel);

        JPanel antialiasPanel = createGridBagPanel("Antialiasing");
        antialiasPanel.add(chkBoxAntialiasStations = new JCheckBox("Stations.", Settings.antialiasing), createGbc(0, 0));
        antialiasPanel.add(chkBoxAntialiasClusters = new JCheckBox("Clusters.", Settings.antialiasingClusters), createGbc(1, 0));
        antialiasPanel.add(chkBoxAntialiasQuakes = new JCheckBox("Earthquakes.", Settings.antialiasingQuakes), createGbc(0, 1));
        antialiasPanel.add(chkBoxAntialiasOldQuakes = new JCheckBox("Archived Earthquakes.", Settings.antialiasingOldQuakes), createGbc(1, 1));

        generalPanel.add(antialiasPanel);

        panel.add(generalPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createEventsTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel eventsPanel = createGridBagPanel("Old events settings");

        chkBoxEnableTimeFilter = new JCheckBox("Don't display older than (hours):", Settings.oldEventsTimeFilterEnabled);
        chkBoxEnableTimeFilter.addChangeListener(changeEvent -> textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected()));
        eventsPanel.add(chkBoxEnableTimeFilter, createGbc(0, 0));

        textFieldTimeFilter = new JTextField(Settings.oldEventsTimeFilter.toString());
        textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected());
        eventsPanel.add(textFieldTimeFilter, createGbc(1, 0));

        chkBoxEnableMagnitudeFilter = new JCheckBox("Don't display smaller than (magnitude):", Settings.oldEventsMagnitudeFilterEnabled);
        chkBoxEnableMagnitudeFilter.addChangeListener(changeEvent -> textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected()));
        eventsPanel.add(chkBoxEnableMagnitudeFilter, createGbc(0, 1));

        textFieldMagnitudeFilter = new JTextField(Settings.oldEventsMagnitudeFilter.toString());
        textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected());
        eventsPanel.add(textFieldMagnitudeFilter, createGbc(1, 1));

        eventsPanel.add(new JLabel("Maximum total number of archived earthquakes:"), createGbc(0, 2));
        eventsPanel.add(textFieldMaxArchived = new JTextField(Settings.maxArchivedQuakes.toString()), createGbc(1, 2));

        JPanel sliderOpacityPanel = createHorizontalPanel();
        sliderOpacityPanel.add(new JLabel("Old events opacity:"));
        sliderOpacity = new JSlider(SwingConstants.HORIZONTAL, 0, 100, Settings.oldEventsOpacity.intValue());
        sliderOpacity.setMajorTickSpacing(10);
        sliderOpacity.setMinorTickSpacing(2);
        sliderOpacity.setPaintTicks(true);
        sliderOpacity.setPaintLabels(true);
        sliderOpacity.setPaintTrack(true);

        sliderOpacity.addChangeListener(changeEvent -> {
            Settings.oldEventsOpacity = (double) sliderOpacity.getValue();
            Settings.changes++;
        });
        sliderOpacityPanel.add(sliderOpacity);
        eventsPanel.add(sliderOpacityPanel, createGbc(3));

        JPanel colorsPanel = new JPanel();
        colorsPanel.setBorder(BorderFactory.createTitledBorder("Old events color"));
        JRadioButton buttonColorByAge = new JRadioButton("Color by age");
        JRadioButton buttonColorByDepth = new JRadioButton("Color by depth");
        JRadioButton buttonColorByMagnitude = new JRadioButton("Color by magnitude");

        colorButtons = new JRadioButton[]{buttonColorByAge, buttonColorByDepth, buttonColorByMagnitude};
        ButtonGroup bg = new ButtonGroup();

        colorButtons[Math.max(0, Math.min(colorButtons.length - 1, Settings.selectedEventColorIndex))].setSelected(true);

        var colorButtonActionListener = new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                for (int i = 0; i < colorButtons.length; i++) {
                    JRadioButton button = colorButtons[i];
                    if (button.isSelected()) {
                        Settings.selectedEventColorIndex = i;
                        break;
                    }
                }
            }
        };

        for (JRadioButton button : colorButtons) {
            bg.add(button);
            button.addActionListener(colorButtonActionListener);
            colorsPanel.add(button);
        }

        eventsPanel.add(colorsPanel, createGbc(4));

        JPanel qualityFilterPanel = createGridBagPanel("Quality");

        qualityFilterPanel.add(new JLabel("Only show old events with quality equal or better than:"), createGbc(0, 0));

        comboBoxQuality = new JComboBox<>(QualityClass.values());
        comboBoxQuality.setSelectedIndex(Math.max(0, Math.min(QualityClass.values().length - 1, Settings.qualityFilter)));
        qualityFilterPanel.add(alignLeft(comboBoxQuality), createGbc(1, 0));

        eventsPanel.add(qualityFilterPanel, createGbc(5));

        panel.add(eventsPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createStationsTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel stationsPanel = createVerticalPanel("Stations");

        JPanel appearancePanel = createGridBagPanel("Appearance");
        appearancePanel.add(chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated).", Settings.useOldColorScheme), createGbc(0, 0));
        appearancePanel.add(chkBoxDeadStations = new JCheckBox("Hide stations with no data.", Settings.hideDeadStations), createGbc(1, 0));
        stationsPanel.add(appearancePanel);

        JPanel stationsShapePanel = new JPanel();
        stationsShapePanel.setBorder(BorderFactory.createTitledBorder("Shape"));

        ButtonGroup buttonGroup = new ButtonGroup();

        JRadioButton buttonCircles = new JRadioButton("Circles");
        JRadioButton buttonTriangles = new JRadioButton("Triangles");
        JRadioButton buttonDepends = new JRadioButton("Based on sensor type");

        JRadioButton[] buttons = new JRadioButton[]{buttonCircles, buttonTriangles, buttonDepends};

        var shapeActionListener = new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                for (int i = 0; i < buttons.length; i++) {
                    JRadioButton button = buttons[i];
                    if (button.isSelected()) {
                        Settings.stationsShapeIndex = i;
                        break;
                    }
                }
            }
        };

        for (JRadioButton button : buttons) {
            buttonGroup.add(button);
            stationsShapePanel.add(button);
            button.addActionListener(shapeActionListener);
        }

        buttons[Settings.stationsShapeIndex].setSelected(true);

        stationsPanel.add(stationsShapePanel);

        JPanel intensityPanel = createGridBagPanel();
        intensityPanel.add(new JLabel("Display station's intensity label at zoom level (0 very close, 200 very far):"), createGbc(0));

        sliderIntensityZoom = new JSlider(SwingConstants.HORIZONTAL, 0, 200, (int) (Settings.stationIntensityVisibilityZoomLevel * 100));
        sliderIntensityZoom.setMajorTickSpacing(10);
        sliderIntensityZoom.setMinorTickSpacing(5);
        sliderIntensityZoom.setPaintTicks(true);
        sliderIntensityZoom.setPaintLabels(true);

        sliderIntensityZoom.addChangeListener(changeEvent -> {
            Settings.stationIntensityVisibilityZoomLevel = sliderIntensityZoom.getValue() / 100.0;
            Settings.changes++;
        });

        intensityPanel.add(sliderIntensityZoom, createGbc(1));
        stationsPanel.add(intensityPanel);

        JPanel stationSizePanel = createGridBagPanel();
        stationSizePanel.add(new JLabel("Stations size multiplier (100 default, 20 tiny, 300 huge):"), createGbc(0));

        sliderStationsSize = new JSlider(SwingConstants.HORIZONTAL, 20, 300, (int) (Settings.stationsSizeMul * 100));
        sliderStationsSize.setMajorTickSpacing(20);
        sliderStationsSize.setMinorTickSpacing(10);
        sliderStationsSize.setPaintTicks(true);
        sliderStationsSize.setPaintLabels(true);

        sliderStationsSize.addChangeListener(changeEvent -> {
            Settings.stationsSizeMul = sliderStationsSize.getValue() / 100.0;
            Settings.changes++;
        });

        stationSizePanel.add(sliderStationsSize, createGbc(1));
        stationsPanel.add(stationSizePanel);

        panel.add(stationsPanel, BorderLayout.NORTH);

        return panel;
    }

    private Component createCinemaModeTab() {
        JPanel panel = new JPanel(new BorderLayout());

        JPanel cinemaModePanel = createGridBagPanel();

        cinemaModePanel.add(new JLabel("Switch to another point of interest after (seconds):"), createGbc(0, 0));
        cinemaModePanel.add(textFieldTime = new JTextField(String.valueOf(Settings.cinemaModeSwitchTime)), createGbc(1, 0));

        JPanel cinemaZoomPanel = createHorizontalPanel();
        cinemaZoomPanel.add(new JLabel("Zoom multiplier (move right to zoom in):"));
        sliderZoomMul = new JSlider(SwingConstants.HORIZONTAL, 20, 500, Settings.cinemaModeZoomMultiplier);
        sliderZoomMul.setMinorTickSpacing(10);
        sliderZoomMul.setMajorTickSpacing(50);
        sliderZoomMul.setPaintTicks(true);
        cinemaZoomPanel.add(sliderZoomMul);
        cinemaModePanel.add(cinemaZoomPanel, createGbc(1));

        cinemaModePanel.add(chkBoxEnableOnStartup = new JCheckBox("Enable Cinema Mode on startup.", Settings.cinemaModeOnStartup), createGbc(0, 2));
        cinemaModePanel.add(chkBoxReEnable = new JCheckBox("Re-enable Cinema Mode automatically.", Settings.cinemaModeReenable), createGbc(1, 2));

        panel.add(cinemaModePanel, BorderLayout.NORTH);
        return panel;
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
        Settings.fpsIdle = sliderFpsIdle.getValue();

        Settings.antialiasing = chkBoxAntialiasStations.isSelected();
        Settings.antialiasingClusters = chkBoxAntialiasClusters.isSelected();
        Settings.antialiasingQuakes = chkBoxAntialiasQuakes.isSelected();
        Settings.antialiasingOldQuakes = chkBoxAntialiasOldQuakes.isSelected();

        Settings.oldEventsTimeFilterEnabled = chkBoxEnableTimeFilter.isSelected();
        Settings.oldEventsTimeFilter = parseDouble(textFieldTimeFilter.getText(), "Old events max age", 0, 24 * 365);

        Settings.oldEventsMagnitudeFilterEnabled = chkBoxEnableMagnitudeFilter.isSelected();
        Settings.oldEventsMagnitudeFilter = parseDouble(textFieldMagnitudeFilter.getText(), "Old events min magnitude", 0, 10);

        Settings.oldEventsOpacity = (double) sliderOpacity.getValue();
        Settings.selectedDateFormatIndex = comboBoxDateFormat.getSelectedIndex();
        Settings.use24HFormat = chkBox24H.isSelected();

        Settings.hideDeadStations = chkBoxDeadStations.isSelected();
        Settings.stationIntensityVisibilityZoomLevel = sliderIntensityZoom.getValue() / 100.0;
        Settings.stationsSizeMul = sliderStationsSize.getValue() / 100.0;

        Settings.maxArchivedQuakes = parseInt(textFieldMaxArchived.getText(), "Max number of archived quakes", 1, Integer.MAX_VALUE);

        Settings.cinemaModeZoomMultiplier = sliderZoomMul.getValue();
        Settings.cinemaModeSwitchTime = parseInt(textFieldTime.getText(), "Cinema mode switch time", 1, 3600);
        Settings.cinemaModeOnStartup = chkBoxEnableOnStartup.isSelected();
        Settings.cinemaModeReenable = chkBoxReEnable.isSelected();

        Settings.displaySystemInfo = chkBoxDisplaySystemInfo.isSelected();
        Settings.displayMagnitudeHistogram = chkBoxDisplayMagnitudeHistogram.isSelected();
        Settings.displayAdditionalQuakeInfo = chkBoxDisplayQuakeAdditionalInfo.isSelected();
        Settings.displayAlertBox = chkBoxAlertBox.isSelected();
        Settings.displayShakemaps = chkBoxShakemap.isSelected();
        Settings.displayTime = chkBoxTime.isSelected();
        Settings.displayCityIntensities = chkBoxCityIntensities.isSelected();
        Settings.displayCapitalCities = chkBoxCapitals.isSelected();

        Settings.qualityFilter = comboBoxQuality.getSelectedIndex();

        Settings.displayClusters = chkBoxClusters.isSelected();
        Settings.displayClusterRoots = chkBoxClusterRoots.isSelected();
        Settings.hideClustersWithQuake = chkBoxHideClusters.isSelected();
    }

    @Override
    public String getTitle() {
        return "Graphics";
    }
}