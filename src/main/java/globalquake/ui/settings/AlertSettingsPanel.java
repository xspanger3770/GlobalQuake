package globalquake.ui.settings;

import javax.swing.*;
import java.awt.*;
import java.text.ParseException;

public class AlertSettingsPanel extends SettingsPanel {

    private JCheckBox chkBoxLocal;
    private JTextField textFieldLocalDist;
    private JCheckBox chkBoxRegion;
    private JTextField textFieldRegionMag;
    private JTextField textFieldRegionDist;
    private JCheckBox checkBoxGlobal;
    private JTextField textFieldGlobalMag;
    private JLabel label1;
    private JCheckBox chkBoxFocus;

    public AlertSettingsPanel() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        createAlertDialogSettings();

        for(int i = 0; i < 10; i++){
            add(new JPanel()); // fillers
        }

        refreshUI();
    }

    private void createAlertDialogSettings() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBorder(BorderFactory.createTitledBorder("Alert dialogs settings"));

        chkBoxLocal = new JCheckBox("", Settings.alertLocal);
        textFieldLocalDist = new JTextField("1", 12);
        textFieldLocalDist.setEnabled(chkBoxLocal.isSelected());
        chkBoxLocal.addChangeListener(changeEvent -> textFieldLocalDist.setEnabled(chkBoxLocal.isSelected()));

        JPanel localPanel = new JPanel(new GridLayout(1,1));
        localPanel.setBorder(BorderFactory.createTitledBorder("Local area"));

        JPanel nearbyPanel = new JPanel();
        nearbyPanel.setLayout(new BoxLayout(nearbyPanel, BoxLayout.X_AXIS));
        nearbyPanel.add(chkBoxLocal);
        nearbyPanel.add(textFieldLocalDist);

        localPanel.add(nearbyPanel);
        panel.add(localPanel);

        chkBoxRegion = new JCheckBox("Show for earthquakes larger than (magnitude): ", Settings.alertRegion);
        textFieldRegionMag = new JTextField(String.valueOf(Settings.alertRegionMag) ,12);
        textFieldRegionMag.setEnabled(chkBoxRegion.isSelected());
        textFieldRegionDist =  new JTextField("1",12);
        textFieldRegionDist.setEnabled(chkBoxRegion.isSelected());

        chkBoxRegion.addChangeListener(changeEvent -> {
            textFieldRegionMag.setEnabled(chkBoxRegion.isSelected());
            textFieldRegionDist.setEnabled(chkBoxRegion.isSelected());
        });

        JPanel regionPanel = new JPanel(new GridLayout(2,1));
        regionPanel.setBorder(BorderFactory.createTitledBorder("Regional area"));

        JPanel regionMagPanel = new JPanel();
        regionMagPanel.setLayout(new BoxLayout(regionMagPanel, BoxLayout.X_AXIS));
        regionMagPanel.add(chkBoxRegion);
        regionMagPanel.add(textFieldRegionMag);

        regionPanel.add(regionMagPanel);


        JPanel regionDistPanel = new JPanel();
        regionDistPanel.setLayout(new BoxLayout(regionDistPanel, BoxLayout.X_AXIS));
        regionDistPanel.add(label1 = new JLabel(""));
        regionDistPanel.add(textFieldRegionDist);

        regionPanel.add(regionDistPanel);

        panel.add(regionPanel);

        JPanel globalPanel = new JPanel(new GridLayout(1,1));
        globalPanel.setBorder(BorderFactory.createTitledBorder("Global"));

        checkBoxGlobal = new JCheckBox("Show for earthquakes larger than (magnitude): ", Settings.alertGlobal);
        textFieldGlobalMag = new JTextField(String.valueOf(Settings.alertGlobalMag), 12);
        textFieldGlobalMag.setEnabled(checkBoxGlobal.isSelected());
        checkBoxGlobal.addChangeListener(changeEvent -> textFieldGlobalMag.setEnabled(checkBoxGlobal.isSelected()));

        JPanel globalMagPanel = new JPanel();
        globalMagPanel.setLayout(new BoxLayout(globalMagPanel, BoxLayout.X_AXIS));

        globalMagPanel.add(checkBoxGlobal);
        globalMagPanel.add(textFieldGlobalMag);

        globalPanel.add(globalMagPanel);

        panel.add(globalPanel);

        JPanel panel2 = new JPanel(new GridLayout(1,2));

        chkBoxFocus = new JCheckBox("Focus main window if the conditions above are met", Settings.focusOnEvent);

        panel2.add(chkBoxFocus);

        panel.add(panel2);

        add(panel);
    }

    @Override
    public void refreshUI() {
        chkBoxLocal.setText("Show when any earthquake occurs closer than (%s): ".formatted(Settings.getSelectedDistanceUnit().getShortName()));
        label1.setText("and are closer from home location than (%s): ".formatted(Settings.getSelectedDistanceUnit().getShortName()));

        textFieldLocalDist.setText(String.format("%.1f", Settings.alertLocalDist * Settings.getSelectedDistanceUnit().getKmRatio()));
        textFieldRegionDist.setText(String.format("%.1f", Settings.alertRegionDist * Settings.getSelectedDistanceUnit().getKmRatio()));

        revalidate();
        repaint();
    }

    @Override
    public void save() throws NumberFormatException, ParseException {
        Settings.alertLocal = chkBoxLocal.isSelected();
        Settings.alertLocalDist = Double.parseDouble(textFieldLocalDist.getText()) / Settings.getSelectedDistanceUnit().getKmRatio();
        Settings.alertRegion = chkBoxRegion.isSelected();
        Settings.alertRegionMag = Double.parseDouble(textFieldRegionMag.getText());
        Settings.alertRegionDist = Double.parseDouble(textFieldRegionDist.getText()) / Settings.getSelectedDistanceUnit().getKmRatio();

        Settings.alertGlobal = checkBoxGlobal.isSelected();
        Settings.alertGlobalMag = Double.parseDouble(textFieldGlobalMag.getText());
        Settings.focusOnEvent = chkBoxFocus.isSelected();
    }

    @Override
    public String getTitle() {
        return "Alerts";
    }
}
