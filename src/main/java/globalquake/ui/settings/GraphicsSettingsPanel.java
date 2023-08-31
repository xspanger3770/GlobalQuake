package globalquake.ui.settings;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;

public class GraphicsSettingsPanel extends SettingsPanel{

    private JCheckBox chkBoxScheme;
    private JCheckBox chkBoxAntialiasing;
    private JSlider sliderFpsIdle;
    private JCheckBox chkBoxEnableTimeFilter;
    private JTextField textFieldTimeFilter;

    private JCheckBox chkBoxEnableMagnitudeFilter;
    private JTextField textFieldMagnitudeFilter;
    private JSlider sliderOpacity;


    public GraphicsSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBorder(new EmptyBorder(5,5,5,5));

        createFpsSlider();
        createEventsSettings();
        createOtherSettings();

        // fillers
        for(int i = 0; i < 10; i++){
            add(new JPanel());
        }
    }

    private void createOtherSettings() {
        JPanel otherSettingsPanel = new JPanel(new GridLayout(2, 1));
        otherSettingsPanel.setBorder(BorderFactory.createTitledBorder("Appearance"));

        chkBoxScheme = new JCheckBox("Use old color scheme (exaggerated)");
        chkBoxScheme.setSelected(Settings.useOldColorScheme);
        otherSettingsPanel.add(chkBoxScheme);

        chkBoxAntialiasing = new JCheckBox("Enable antialiasing for stations");
        chkBoxAntialiasing.setSelected(Settings.antialiasing);
        otherSettingsPanel.add(chkBoxAntialiasing);

        add(otherSettingsPanel);
    }

    private void createEventsSettings() {
        JPanel eventsPanel = new JPanel();
        eventsPanel.setBorder(BorderFactory.createTitledBorder("Old events settings"));
        eventsPanel.setLayout(new BoxLayout(eventsPanel, BoxLayout.Y_AXIS));

        JPanel timePanel = new JPanel();
        timePanel.setLayout(new BoxLayout(timePanel, BoxLayout.X_AXIS));
        timePanel.setBorder(new EmptyBorder(5,5,5,5));

        chkBoxEnableTimeFilter = new JCheckBox("Don't display events older than (hours): ");
        chkBoxEnableTimeFilter.setSelected(Settings.oldEventsTimeFilterEnabled);

        textFieldTimeFilter = new JTextField(Settings.oldEventsTimeFilter.toString(), 12);
        textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected());

        chkBoxEnableTimeFilter.addChangeListener(changeEvent -> textFieldTimeFilter.setEnabled(chkBoxEnableTimeFilter.isSelected()));

        timePanel.add(chkBoxEnableTimeFilter);
        timePanel.add((textFieldTimeFilter));

        eventsPanel.add(timePanel);

        JPanel magnitudePanel = new JPanel();
        magnitudePanel.setBorder(new EmptyBorder(5,5,5,5));
        magnitudePanel.setLayout(new BoxLayout(magnitudePanel, BoxLayout.X_AXIS));

        chkBoxEnableMagnitudeFilter = new JCheckBox("Don't display events smaller than (magnitude): ");
        chkBoxEnableMagnitudeFilter.setSelected(Settings.oldEventsMagnitudeFilterEnabled);

        textFieldMagnitudeFilter = new JTextField(Settings.oldEventsMagnitudeFilter.toString(), 12);
        textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected());

        chkBoxEnableMagnitudeFilter.addChangeListener(changeEvent -> textFieldMagnitudeFilter.setEnabled(chkBoxEnableMagnitudeFilter.isSelected()));

        magnitudePanel.add(chkBoxEnableMagnitudeFilter);
        magnitudePanel.add((textFieldMagnitudeFilter));

        eventsPanel.add(magnitudePanel);

        JPanel opacityPanel = new JPanel();
        opacityPanel.setBorder(new EmptyBorder(5,5,5,5));
        opacityPanel.setLayout(new BoxLayout(opacityPanel, BoxLayout.X_AXIS));

        sliderOpacity = new JSlider(JSlider.HORIZONTAL,0,100, Settings.oldEventsOpacity.intValue());
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

        add(eventsPanel);
    }

    private void createFpsSlider() {
        JPanel fpsPanel = new JPanel();
        fpsPanel.setBorder(BorderFactory.createTitledBorder("Performance"));
        fpsPanel.setLayout(new BoxLayout(fpsPanel, BoxLayout.Y_AXIS));

        sliderFpsIdle = new JSlider(JSlider.HORIZONTAL, 10, 90, Settings.fpsIdle);
        sliderFpsIdle.setPaintLabels(true);
        sliderFpsIdle.setPaintTicks(true);
        sliderFpsIdle.setMajorTickSpacing(10);
        sliderFpsIdle.setMinorTickSpacing(5);
        sliderFpsIdle.setBorder(new EmptyBorder(5,5,10,5));

        JLabel label = new JLabel("FPS at idle: "+sliderFpsIdle.getValue());

        sliderFpsIdle.addChangeListener(changeEvent -> label.setText("FPS at idle: "+sliderFpsIdle.getValue()));

        fpsPanel.add(label);
        fpsPanel.add(sliderFpsIdle);

        add(fpsPanel);
    }

    @Override
    public void save() {
        Settings.useOldColorScheme = chkBoxScheme.isSelected();
        Settings.antialiasing = chkBoxAntialiasing.isSelected();
        Settings.fpsIdle = sliderFpsIdle.getValue();

        Settings.oldEventsTimeFilterEnabled = chkBoxEnableTimeFilter.isSelected();
        Settings.oldEventsTimeFilter = Double.parseDouble(textFieldTimeFilter.getText());

        Settings.oldEventsMagnitudeFilterEnabled = chkBoxEnableMagnitudeFilter.isSelected();
        Settings.oldEventsMagnitudeFilter = Double.parseDouble(textFieldMagnitudeFilter.getText());

        Settings.oldEventsOpacity = (double) sliderOpacity.getValue();
    }

    @Override
    public String getTitle() {
        return "Graphics";
    }
}
