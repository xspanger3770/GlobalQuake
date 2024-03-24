package globalquake.ui.settings;


import globalquake.core.Settings;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScale;
import globalquake.core.intensity.IntensityScales;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.time.Instant;
import java.time.ZoneId;
import java.util.Comparator;
import java.util.Objects;

public class GeneralSettingsPanel extends SettingsPanel {
    private JComboBox<IntensityScale> comboBoxScale;
    private JCheckBox chkBoxHomeLoc;

    private JTextField textFieldLat;
    private JTextField textFieldLon;
    private JComboBox<DistanceUnit> distanceUnitJComboBox;
    private JComboBox<ZoneId> timezoneCombobox;

    private JSlider sliderStoreTime;

    public GeneralSettingsPanel(SettingsFrame settingsFrame) {
        setLayout(new BorderLayout());

        JPanel generalPanel = createVerticalPanel(false);
        generalPanel.add(createHomeLocationSettings());
        generalPanel.add(createIntensitySettingsPanel());
        generalPanel.add(createOtherSettings(settingsFrame));
        generalPanel.add(createSettingStoreTime());

        add(generalPanel, BorderLayout.NORTH);
    }

    private Component createHomeLocationSettings() {
        JPanel outsidePanel = createGridBagPanel("Home location settings");

        outsidePanel.add(new JLabel("Home Latitude:"), createGbc(0, 0));
        outsidePanel.add(textFieldLat = new JTextField(String.format("%s", Settings.homeLat)), createGbc(1, 0));

        outsidePanel.add(new JLabel("Home Longitude:"), createGbc(0, 1));
        outsidePanel.add(textFieldLon = new JTextField(String.format("%s", Settings.homeLon)), createGbc(1, 1));

        outsidePanel.add(createJTextArea("Home location will be used for playing additional alarm sounds if an earthquake occurs nearby.", outsidePanel), createGbc(2));

        outsidePanel.add(chkBoxHomeLoc = new JCheckBox("Display home location.", Settings.displayHomeLocation), createGbc(3));

        return outsidePanel;
    }

    private Component createIntensitySettingsPanel() {
        JPanel panel = createGridBagPanel("Intensity scale");

        comboBoxScale = new JComboBox<>(IntensityScales.INTENSITY_SCALES);
        comboBoxScale.setSelectedIndex(Settings.intensityScaleIndex);
        panel.add(alignLeft(comboBoxScale), createGbc(0));

        panel.add(createJTextArea("Keep in mind that the displayed intensities are estimated, not measured.", panel), createGbc(1));

        return panel;
    }

    private Component createOtherSettings(SettingsFrame settingsFrame) {
        JPanel panel = createGridBagPanel("Other");

        panel.add(new JLabel("Distance units:"), createGbc(0, 0));
        distanceUnitJComboBox = new JComboBox<>(DistanceUnit.values());
        distanceUnitJComboBox.setSelectedIndex(Math.max(0, Math.min(distanceUnitJComboBox.getItemCount() - 1, Settings.distanceUnitsIndex)));
        distanceUnitJComboBox.addItemListener(itemEvent -> {
            Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
            settingsFrame.refreshUI();
        });
        panel.add(alignLeft(distanceUnitJComboBox), createGbc(1, 0));

        panel.add(new JLabel("Timezone:"), createGbc(0, 1));
        Comparator<ZoneId> zoneIdComparator = Comparator.comparingInt(zone -> zone.getRules().getOffset(Instant.now()).getTotalSeconds());

        // Use a DefaultComboBoxModel for better control and management
        DefaultComboBoxModel<ZoneId> timezoneModel = new DefaultComboBoxModel<>();

        // Populate the model with available timezones and sort them using the custom Comparator
        ZoneId.getAvailableZoneIds().stream()
                .map(ZoneId::of)
                .sorted(zoneIdComparator)
                .forEach(timezoneModel::addElement);

        // Create the JComboBox with the populated and sorted model
        timezoneCombobox = new JComboBox<>(timezoneModel);
        // this assures that default timezone will always be selected
        timezoneCombobox.setSelectedItem(ZoneId.systemDefault());

        // if there's a valid timezone in the settings, then it will be selected
        timezoneCombobox.setSelectedItem(ZoneId.of(Settings.timezoneStr));

        timezoneCombobox.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                JLabel label = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

                if (value instanceof ZoneId zoneId) {
                    String offset = zoneId.getRules().getOffset(Instant.now()).toString();
                    label.setText(String.format("%s (%s)", zoneId, offset));
                }

                return label;
            }
        });

        timezoneCombobox.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                Settings.timezoneStr = ((ZoneId) Objects.requireNonNull(timezoneCombobox.getSelectedItem())).getId();
                Settings.initTimezoneSettings();
            }
        });

        panel.add(alignLeft(timezoneCombobox), createGbc(1, 1));

        return panel;
    }

    private Component createSettingStoreTime() {
        sliderStoreTime = HypocenterAnalysisSettingsPanel.createSettingsSlider(2, 20, 2, 1);

        JLabel label = new JLabel("Waveform data storage time (minutes): %d".formatted(sliderStoreTime.getValue()));
        sliderStoreTime.addChangeListener(e -> label.setText("Waveform data storage time (minutes): %d".formatted(sliderStoreTime.getValue())));
        sliderStoreTime.setValue(Settings.logsStoreTimeMinutes);

        return HypocenterAnalysisSettingsPanel.createCoolLayout(sliderStoreTime, label, "5",
                """
                        In GlobalQuake, waveform data poses the highest demand on your system's RAM.
                        If you're encountering memory constraints, you have two options:
                        either reduce the number of selected stations or lower this specific value.
                        """);
    }

    @Override
    public void save() {
        Settings.homeLat = parseDouble(textFieldLat.getText(), "Home latitude", -90, 90);
        Settings.homeLon = parseDouble(textFieldLon.getText(), "Home longitude", -180, 180);
        Settings.intensityScaleIndex = comboBoxScale.getSelectedIndex();
        Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
        Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
        Settings.timezoneStr = ((ZoneId) Objects.requireNonNull(timezoneCombobox.getSelectedItem())).getId();
        Settings.logsStoreTimeMinutes = sliderStoreTime.getValue();
    }

    @Override
    public String getTitle() {
        return "General";
    }
}