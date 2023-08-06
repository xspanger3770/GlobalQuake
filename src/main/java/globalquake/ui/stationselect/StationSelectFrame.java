package globalquake.ui.stationselect;

import globalquake.ui.database.DatabaseMonitorFrame;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.util.Timer;
import java.util.TimerTask;

public class StationSelectFrame extends JFrame {

    private final StationSelectPanel stationSelectPanel;

    public StationSelectFrame(DatabaseMonitorFrame owner) {
        setLayout(new BorderLayout());

        stationSelectPanel = new StationSelectPanel(owner.getManager().getStationDatabase());

        setPreferredSize(new Dimension(1000, 800));

        add(stationSelectPanel, BorderLayout.CENTER);
        add(createControlPanel(), BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setResizable(true);
        setTitle("Select Stations");

        java.util.Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                stationSelectPanel.repaint();
            }
        }, 0, 1000 / 40);
    }

    private JPanel createControlPanel() {
        JPanel controlPanel = new JPanel();

        JCheckBox chkBoxShowUnavailable = new JCheckBox("Show not available stations");
        chkBoxShowUnavailable.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                System.out.println("AASDADS");
                stationSelectPanel.showUnavailable = chkBoxShowUnavailable.isSelected();
                stationSelectPanel.updateAllStations();
            }
        });

        controlPanel.add(chkBoxShowUnavailable);

        return controlPanel;
    }
}
