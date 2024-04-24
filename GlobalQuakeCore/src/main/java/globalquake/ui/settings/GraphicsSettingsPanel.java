package globalquake.ui.settings;

import globalquake.core.Settings;
import globalquake.core.earthquake.quality.QualityClass;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
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

    private Component createCinemaModeTab() {
        JPanel panel = new JPanel();
        panel.setBorder(new EmptyBorder(6, 6, 6, 6));
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        textFieldTime = new JTextField(String.valueOf(Settings.cinemaModeSwitchTime), 12);

        JPanel timePanel = new JPanel();
        timePanel.setLayout(new BoxLayout(timePanel, BoxLayout.X_AXIS));
        timePanel.add(new JLabel("Switch to another point of interest after (seconds): "));
        timePanel.add(textFieldTime);
        panel.add(timePanel);

        JPanel zoomPanel = new JPanel();
        zoomPanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        zoomPanel.setLayout(new BoxLayout(zoomPanel, BoxLayout.X_AXIS));
        zoomPanel.add(new JLabel("Zoom multiplier (move right to zoom in):"));

        sliderZoomMul = new JSlider(JSlider.HORIZONTAL, 20, 500, Settings.cinemaModeZoomMultiplier);
        sliderZoomMul.setMinorTickSpacing(10);
        sliderZoomMul.setMajorTickSpacing(50);
        sliderZoomMul.setPaintTicks(true);

        zoomPanel.add(sliderZoomMul);
        panel.add(zoomPanel);

        JPanel checkboxPanel = new JPanel();

        checkboxPanel.add(chkBoxEnableOnStartup = new JCheckBox("Enable Cinema Mode on startup", Settings.cinemaModeOnStartup));
        checkboxPanel.add(chkBoxReEnable = new JCheckBox("Re-enable Cinema Mode automatically", Settings.cinemaModeReenable));
        panel.add(checkboxPanel);

        fill(panel, 32);

        return panel;
    }

    private Component createGeneralTab() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JPanel performancePanel = new JPanel();
        performancePanel.setLayout(new BoxLayout(performancePanel, BoxLayout.Y_AXIS));
        performancePanel.setBorder(BorderFactory.createTitledBorder("Performance"));

        sliderFpsIdle = new JSlider(JSlider.HORIZONTAL, 10, 200, Settings.fpsIdle);
        sliderFpsIdle.setPaintLabels(true);
        sliderFpsIdle.setPaintTicks(true);
        sliderFpsIdle.setMajorTickSpacing(10);
        sliderFpsIdle.setMinorTickSpacing(5);
        sliderFpsIdle.setBorder(new EmptyBorder(5, 5, 10, 5));

        JLabel label = new JLabel("FPS limit: " + sliderFpsIdle.getValue());

        sliderFpsIdle.addChangeListener(changeEvent -> label.setText("FPS limit: " + sliderFpsIdle.getValue()));

        performancePanel.add(label);
        performancePanel.add(sliderFpsIdle);

        panel.add(performancePanel);

        JPanel dateFormatPanel = new JPanel();
        dateFormatPanel.setBorder(BorderFactory.createTitledBorder("Date and Time setting"));

        comboBoxDateFormat = new JComboBox<>();
        Instant now = Instant.now();
        for (DateTimeFormatter formatter : Settings.DATE_FORMATS) {
            comboBoxDateFormat.addItem(formatter.format(now));
        }

        comboBoxDateFormat.setSelectedIndex(Settings.selectedDateFormatIndex);

        dateFormatPanel.add(new JLabel("Preferred date format: "));
        dateFormatPanel.add(comboBoxDateFormat);
        dateFormatPanel.add(chkBox24H = new JCheckBox("Use 24 hours format", Settings.use24HFormat));

        panel.add(dateFormatPanel);

        JPanel mainWindowPanel = new JPanel(new GridLayout(4, 2));
        mainWindowPanel.setBorder(new TitledBorder("Main Screen"));

        mainWindowPanel.add(chkBoxDisplaySystemInfo = new JCheckBox("Display system info", Settings.displaySystemInfo));
        mainWindowPanel.add(chkBoxDisplayMagnitudeHistogram = new JCheckBox("Display magnitude histogram", Settings.displayMagnitudeHistogram));
        mainWindowPanel.add(chkBoxDisplayQuakeAdditionalInfo = new JCheckBox("Display technical earthquake data", Settings.displayAdditionalQuakeInfo));
        mainWindowPanel.add(chkBoxAlertBox = new JCheckBox("Display alert box for nearby earthquakes", Settings.displayAlertBox));
        mainWindowPanel.add(chkBoxShakemap = new JCheckBox("Display shakemap hexagons", Settings.displayShakemaps));
        mainWindowPanel.add(chkBoxTime = new JCheckBox("Display time", Settings.displayTime));
        mainWindowPanel.add(chkBoxCityIntensities = new JCheckBox("Display estimated intensities in cities", Settings.displayCityIntensities));
        mainWindowPanel.add(chkBoxCapitals = new JCheckBox("Display capital cities", Settings.displayCapitalCities));

        panel.add(mainWindowPanel);

        JPanel clustersPanel = new JPanel(new GridLayout(3, 1));
        clustersPanel.setBorder(new TitledBorder("Cluster settings"));

        clustersPanel.add(chkBoxClusterRoots = new JCheckBox("Display Clusters (possible shaking locations)", Settings.displayClusterRoots));
        clustersPanel.add(chkBoxClusters = new JCheckBox("Display Stations assigned to Clusters (local mode only)", Settings.displayClusters));
        clustersPanel.add(chkBoxHideClusters = new JCheckBox("Hide Cluster after the Earthquake is actually found", Settings.hideClustersWithQuake));

        panel.add(clustersPanel);

        JPanel antialiasPanel = new JPanel(new GridLayout(3, 1));
        antialiasPanel.setBorder(new TitledBorder("Antialiasing"));
        antialiasPanel.add(chkBoxAntialiasStations = new JCheckBox("Stations", Settings.antialiasing));
        antialiasPanel.add(chkBoxAntialiasClusters = new JCheckBox("Clusters", Settings.antialiasingClusters));
        antialiasPanel.add(chkBoxAntialiasQuakes = new JCheckBox("Earthquakes", Settings.antialiasingQuakes));
        antialiasPanel.add(chkBoxAntialiasOldQuakes = new JCheckBox("Archived Earthquakes", Settings.antialiasingOldQuakes));

        panel.add(antialiasPanel);

        return panel;
    }

    private Component createEventsTab() {
        JPanel eventsPanel = new JPanel();
        eventsPanel.setBorder(BorderFactory.createTitledBorder("Old events settings"));
        eventsPanel.setLayout(new BoxLayout(eventsPanel, BoxLayout.Y_AXIS));

        JPanel timePanel = new JPanel();
        timePanel.setLayout(new BoxLayout(timePanel, BoxLayout.X_AXIS));
        timePanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        chkBoxEnableTimeFilter = new JCheckBox("Don't display older than (hours): ");
        chkBoxEnableTimeFilter.setSelected(Settings.oldEventsTimeFilterEnabled);

        textFieldTimeFilter = new JTextField(Settings.oldEventsTimeFilter.toString(), 12);
        textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected());

        chkBoxEnableTimeFilter.addChangeListener(changeEvent -> textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected()));

        timePanel.add(chkBoxEnableTimeFilter);
        timePanel.add((textFieldTimeFilter));

        eventsPanel.add(timePanel);

        JPanel magnitudePanel = new JPanel();
        magnitudePanel.setBorder(new EmptyBorder(5, 5, 5, 5));
        magnitudePanel.setLayout(new BoxLayout(magnitudePanel, BoxLayout.X_AXIS));

        chkBoxEnableMagnitudeFilter = new JCheckBox("Don't display smaller than (magnitude): ");
        chkBoxEnableMagnitudeFilter.setSelected(Settings.oldEventsMagnitudeFilterEnabled);

        textFieldMagnitudeFilter = new JTextField(Settings.oldEventsMagnitudeFilter.toString(), 12);
        textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected());

        chkBoxEnableMagnitudeFilter.addChangeListener(changeEvent -> textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected()));

        magnitudePanel.add(chkBoxEnableMagnitudeFilter);
        magnitudePanel.add((textFieldMagnitudeFilter));

        eventsPanel.add(magnitudePanel);

        JPanel removeOldPanel = new JPanel();
        removeOldPanel.setLayout(new BoxLayout(removeOldPanel, BoxLayout.X_AXIS));
        removeOldPanel.setBorder(new EmptyBorder(5, 5, 5, 5));

        textFieldMaxArchived = new JTextField(Settings.maxArchivedQuakes.toString(), 12);

        removeOldPanel.add(new JLabel("Maximum total number of archived earthquakes: "));
        removeOldPanel.add(textFieldMaxArchived);

        eventsPanel.add(removeOldPanel);


        JPanel opacityPanel = new JPanel();
        opacityPanel.setBorder(new EmptyBorder(5, 5, 5, 5));
        opacityPanel.setLayout(new BoxLayout(opacityPanel, BoxLayout.X_AXIS));

        sliderOpacity = new JSlider(JSlider.HORIZONTAL, 0, 100, Settings.oldEventsOpacity.intValue());
        sliderOpacity.setMajorTickSpacing(10);
        sliderOpacity.setMinorTickSpacing(2);
        sliderOpacity.setPaintTicks(true);
        sliderOpacity.setPaintLabels(true);
        sliderOpacity.setPaintTrack(true);

        sliderOpacity.addChangeListener(changeEvent -> {
            Settings.oldEventsOpacity = (double) sliderOpacity.getValue();
            Settings.changes++;
        });

        opacityPanel.add(new JLabel("Old events opacity: "));
        opacityPanel.add(sliderOpacity);

        eventsPanel.add(opacityPanel);

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

        eventsPanel.add(colorsPanel);

        JPanel qualityFilterPanel = new JPanel();
        qualityFilterPanel.setBorder(BorderFactory.createTitledBorder("Quality"));

        qualityFilterPanel.add(new JLabel("Only show old events with quality equal or better than: "));

        comboBoxQuality = new JComboBox<>(QualityClass.values());
        comboBoxQuality.setSelectedIndex(Math.max(0, Math.min(QualityClass.values().length - 1, Settings.qualityFilter)));
        qualityFilterPanel.add(comboBoxQuality);

        eventsPanel.add(qualityFilterPanel);

        fill(eventsPanel, 12);

        return eventsPanel;
    }

    private Component createStationsTab() {
        JPanel stationsPanel = new JPanel();
        stationsPanel.setLayout(new BoxLayout(stationsPanel, BoxLayout.Y_AXIS));
        stationsPanel.setBorder(BorderFactory.createTitledBorder("Stations"));

        JPanel checkBoxes = new JPanel(new GridLayout(1, 2));
        checkBoxes.setBorder(BorderFactory.createTitledBorder("Appearance"));

        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        checkBoxes.add(chkBoxScheme);

        checkBoxes.add(chkBoxDeadStations = new JCheckBox("Hide stations with no data", Settings.hideDeadStations));

        stationsPanel.add(checkBoxes);

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

        JPanel intensityPanel = new JPanel(new GridLayout(2, 1));
        intensityPanel.add(new JLabel("Display station's intensity label at zoom level (0 very close, 200 very far):"));

        sliderIntensityZoom = new JSlider(SwingConstants.HORIZONTAL, 0, 200, (int) (Settings.stationIntensityVisibilityZoomLevel * 100));
        sliderIntensityZoom.setMajorTickSpacing(10);
        sliderIntensityZoom.setMinorTickSpacing(5);
        sliderIntensityZoom.setPaintTicks(true);
        sliderIntensityZoom.setPaintLabels(true);

        sliderIntensityZoom.addChangeListener(changeEvent -> {
            Settings.stationIntensityVisibilityZoomLevel = sliderIntensityZoom.getValue() / 100.0;
            Settings.changes++;
        });

        intensityPanel.add(sliderIntensityZoom);
        stationsPanel.add(intensityPanel);

        JPanel stationSizePanel = new JPanel(new GridLayout(2, 1));
        stationSizePanel.add(new JLabel("Stations size multiplier (100 default, 20 tiny, 300 huge):"));

        sliderStationsSize = new JSlider(SwingConstants.HORIZONTAL, 20, 300, (int) (Settings.stationsSizeMul * 100));
        sliderStationsSize.setMajorTickSpacing(20);
        sliderStationsSize.setMinorTickSpacing(10);
        sliderStationsSize.setPaintTicks(true);
        sliderStationsSize.setPaintLabels(true);

        sliderStationsSize.addChangeListener(changeEvent -> {
            Settings.stationsSizeMul = sliderStationsSize.getValue() / 100.0;
            Settings.changes++;
        });

        stationSizePanel.add(sliderStationsSize);
        stationsPanel.add(stationSizePanel);

        fill(stationsPanel, 6);

        return stationsPanel;
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
