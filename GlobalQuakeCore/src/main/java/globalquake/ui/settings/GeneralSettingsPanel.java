package globalquake.ui.settings;


import globalquake.core.Settings;
import globalquake.core.geo.DistanceUnit;
import globalquake.core.intensity.IntensityScale;
import globalquake.core.intensity.IntensityScales;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.TextStyle;
import java.util.Comparator;
import java.util.Locale;
import java.util.Set;
import java.util.TimeZone;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class GeneralSettingsPanel extends SettingsPanel {
    private JComboBox<IntensityScale> comboBoxScale;
    private JCheckBox chkBoxHomeLoc;

    private double homeLat;
    private double homeLon;

    private JTextField textFieldCoords;
    private JComboBox<DistanceUnit> distanceUnitJComboBox;
    private JComboBox<ZoneId> timezoneCombobox;

    public GeneralSettingsPanel(SettingsFrame settingsFrame) {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));

        createHomeLocationSettings();
        //createAlertsDialogSettings();
        add(createIntensitySettingsPanel());
        createOtherSettings(settingsFrame);

        fill(this, 12);
    }

    private void createOtherSettings(SettingsFrame settingsFrame) {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder("Other"));

        JPanel row1 = new JPanel();

        row1.add(new JLabel("Distance units: "));

        distanceUnitJComboBox = new JComboBox<>(DistanceUnit.values());
        distanceUnitJComboBox.setSelectedIndex(Math.max(0, Math.min(distanceUnitJComboBox.getItemCount() - 1, Settings.distanceUnitsIndex)));

        distanceUnitJComboBox.addItemListener(itemEvent -> {
            Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
            settingsFrame.refreshUI();
        });

        row1.add(distanceUnitJComboBox);

        JPanel row2 = new JPanel();

        row2.add(new JLabel("Timezone: "));
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

        // if theres valid timezone in the settings then it will be selected
        timezoneCombobox.setSelectedItem(ZoneId.of(Settings.timezoneStr));

        // Add the JComboBox to your UI
        row2.add(timezoneCombobox);

        timezoneCombobox.setRenderer(new DefaultListCellRenderer(){
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index,
                                                          boolean isSelected, boolean cellHasFocus) {
                JLabel label = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

                if (value instanceof ZoneId) {
                    ZoneId zoneId = (ZoneId) value;

                    String offset = zoneId.getRules().getOffset(Instant.now()).toString();

                    label.setText(String.format("%s (%s)", zoneId, offset));
                }

                return label;
            }
        });

        timezoneCombobox.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                Settings.timezoneStr = ((ZoneId)timezoneCombobox.getSelectedItem()).getId();
                Settings.initTimezoneSettings();
            }
        });

        row2.add(timezoneCombobox);

        panel.add(row1);
        panel.add(row2);

        add(panel);
    }

    private void createHomeLocationSettings() {
        JPanel outsidePanel = new JPanel(new BorderLayout());
        outsidePanel.setBorder(BorderFactory.createTitledBorder("Home location settings"));

        JPanel homeLocationPanel = new JPanel();
        homeLocationPanel.setLayout(new GridLayout(2,1));

        // This is for the google earth coordinates
        JLabel glgECoords = new JLabel("Home Coordinates: ");
        textFieldCoords = new JTextField(30);
        textFieldCoords.setText(convertToGoogleEarthFormat(Settings.homeLat, Settings.homeLon));
        textFieldCoords.setColumns(20);

        // Creates a panel for entering home coordinates
        JPanel allPanel = new JPanel();
        allPanel.add(glgECoords);
        allPanel.add(textFieldCoords);
        // adds new panel to the home location panel
        homeLocationPanel.add(allPanel);

        // displays information about the home location
        JTextArea infoLocation = new JTextArea
        (
            " * Paste coordinates from google earth to the above label.\n\n" +
            " * Home location will be used for playing additional alarm\n" +
            "   sounds if an earthquake occurs nearby."
        );
        // initializes info text attribs
        infoLocation.setBorder(new EmptyBorder(5,5,5,5));
        infoLocation.setLineWrap(true);
        infoLocation.setEditable(false);
        infoLocation.setBackground(homeLocationPanel.getBackground());

        // create panel for display home loc check box
        chkBoxHomeLoc = new JCheckBox("Display home location");
        chkBoxHomeLoc.setSelected(Settings.displayHomeLocation);
        // add all panels to outside panel
        outsidePanel.add(chkBoxHomeLoc);
        outsidePanel.add(homeLocationPanel, BorderLayout.NORTH);
        outsidePanel.add(infoLocation, BorderLayout.CENTER);
        outsidePanel.add(chkBoxHomeLoc, BorderLayout.SOUTH);
        // add the outside panel
        add(outsidePanel);
    }

    private JPanel createIntensitySettingsPanel() {
        JPanel panel = new JPanel(new GridLayout(2,1));
        panel.setBorder(BorderFactory.createTitledBorder("Intensity Scale"));

        comboBoxScale = new JComboBox<>(IntensityScales.INTENSITY_SCALES);
        comboBoxScale.setSelectedIndex(Settings.intensityScaleIndex);

        JPanel div = new JPanel();
        div.add(comboBoxScale);
        panel.add(div, BorderLayout.CENTER);

        JLabel lbl = new JLabel();
        lbl.setFont(new Font("Calibri", Font.PLAIN, 13));
        lbl.setText("Keep in mind that the displayed intensities are estimated, not measured");


        panel.add(lbl, BorderLayout.SOUTH);

        return panel;
    }

    @Override
    public void save() {
        handleGoogleEarthCoordinates();
        Settings.homeLat = homeLat;
        Settings.homeLon = homeLon;
        Settings.intensityScaleIndex = comboBoxScale.getSelectedIndex();
        Settings.displayHomeLocation = chkBoxHomeLoc.isSelected();
        Settings.distanceUnitsIndex = distanceUnitJComboBox.getSelectedIndex();
        Settings.timezoneStr = ((ZoneId)timezoneCombobox.getSelectedItem()).getId();
    }

    @Override
    public String getTitle() {
        return "General";
    }

    /**
        * @author Chris Eberle
        * @return parses coordinates in google earth format to two
        *         doubles representing lattitude and longitude
    */
    private void handleGoogleEarthCoordinates() {
        // Parse Google Earth coordinates and set homeLat and homeLon accordingly
        try {
            // Regular expression pattern for extracting latitude and longitude from the Google Earth coordinate string
            String regex = "(\\d+)°(\\d+)'(\\d+)\"([NS]) (\\d+)°(\\d+)'(\\d+)\"([EW])";
            Pattern pattern = Pattern.compile(regex);
            Matcher matcher = pattern.matcher(textFieldCoords.getText()); // Use the text directly from the textFieldAll
    
            if (matcher.find()) {
                double latDegrees = Double.parseDouble(matcher.group(1));
                double latMinutes = Double.parseDouble(matcher.group(2));
                double latSeconds = Double.parseDouble(matcher.group(3));
                String latDirection = matcher.group(4);
                double lonDegrees = Double.parseDouble(matcher.group(5));
                double lonMinutes = Double.parseDouble(matcher.group(6));
                double lonSeconds = Double.parseDouble(matcher.group(7));
                String lonDirection = matcher.group(8);
    
                // Convert degrees, minutes, and seconds to decimal degrees
                homeLat = latDegrees + (latMinutes / 60.0) + (latSeconds / 3600.0);
                homeLon = lonDegrees + (lonMinutes / 60.0) + (lonSeconds / 3600.0);
    
                // Adjust for North/South and East/West directions
                if (latDirection.equals("S")) {
                    homeLat = -homeLat;
                }
                if (lonDirection.equals("W")) {
                    homeLon = -homeLon;
                }
    
                // Update the text field with the formatted coordinates
                textFieldCoords.setText(String.format("%.6f %.6f", homeLat, homeLon));
            } else {
                JOptionPane.showMessageDialog(this, "Invalid Google Earth coordinates format.  Defaulting Location to [ 0°00'00\"N 0°00'00\"E ]", "Error", JOptionPane.ERROR_MESSAGE);
            }
        } catch (NumberFormatException e) {
            JOptionPane.showMessageDialog(this, "Error parsing coordinates", "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    /**
        * @author Chris Eberle
        * @param longitude and latitude doubles from settings object
        * @return google earth formatted coordinate string
    */
    private String convertToGoogleEarthFormat(double latitude, double longitude) {
        //initialize formate strings
        String formattedLat;
        String formattedLon;
        // Format latitude
        if ( latitude >= 0){
            formattedLat = formatCoordinate(latitude, "N");
        }else{
            formattedLat = formatCoordinate(latitude*-1, "S");
        }
        // Format longitude
        if ( longitude >= 0){
            formattedLon = formatCoordinate(longitude, "E");
        }
        else{
            formattedLon = formatCoordinate(longitude*-1, "W");
        }
        // Combine and return the formatted coordinates
        return formattedLat + " " + formattedLon;
    }
    
    /**
        * @author Chris Eberle
        * @param double and a String indicating direction(N,S,W,E)
        * @return converts a single lat or long coordinate to the 
        *         correct format.
    */
    private String formatCoordinate(double value, String indicator) {
        int degrees = (int) value;
        int minutes = (int) ((value - degrees) * 60);
        int seconds = (int) ((value - degrees - minutes / 60.0) * 3600);
    
        return String.format("%d°%02d'%02d\"%s", degrees, minutes, seconds, indicator);
    }
    
}

