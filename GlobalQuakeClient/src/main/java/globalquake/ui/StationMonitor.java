package globalquake.ui;

import globalquake.client.GlobalQuakeLocal;
import globalquake.core.station.AbstractStation;
import globalquake.core.station.GlobalStation;
import globalquake.events.specific.StationMonitorCloseEvent;
import globalquake.events.specific.StationMonitorOpenEvent;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Timer;
import java.util.TimerTask;

public class StationMonitor extends GQFrame {

    private final StationMonitorPanel stationMonitorPanel;
    private AbstractStation station;

    public StationMonitor(Component parent, AbstractStation station, int refreshTime) {
        this.station = station;

        if (GlobalQuakeLocal.instance != null && station instanceof GlobalStation globalStation) {
            GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new StationMonitorOpenEvent(StationMonitor.this, globalStation));
        }

        setLayout(new BorderLayout());

        add(createControlPanel(), BorderLayout.NORTH);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        stationMonitorPanel = new StationMonitorPanel(station);
        add(stationMonitorPanel, BorderLayout.CENTER);

        pack();

        setLocationRelativeTo(parent);
        setResizable(true);
        setTitle("Station Monitor - " + station.getNetworkCode() + " " + station.getStationCode() + " "
                + station.getChannelName() + " " + station.getLocationCode());

        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                stationMonitorPanel.updateImage();
                stationMonitorPanel.repaint();
            }
        }, 0, refreshTime);

        addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosed(WindowEvent e) {
                timer.cancel();
                if (GlobalQuakeLocal.instance != null && station instanceof GlobalStation globalStation) {
                    GlobalQuakeLocal.instance.getLocalEventHandler().fireEvent(new StationMonitorCloseEvent(StationMonitor.this, globalStation));
                }
            }
        });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                System.err.println("K");
                if (e.getKeyCode() == KeyEvent.VK_ESCAPE) {
                    System.err.println("ESC");
                    dispose();
                }
            }
        });

        stationMonitorPanel.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                stationMonitorPanel.updateImage();
                stationMonitorPanel.repaint();
            }
        });

        setVisible(true);
    }

    private Component createControlPanel() {
        JPanel panel = new JPanel();

        JCheckBox chkBoxDisable = new JCheckBox("Disable event picking", station.disabled);
        chkBoxDisable.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent actionEvent) {
                station.disabled = chkBoxDisable.isSelected();
            }
        });

        panel.add(chkBoxDisable);

        return panel;
    }

    public AbstractStation getStation() {
        return station;
    }

    public void swapStation(GlobalStation station) {
        this.station = station;
        stationMonitorPanel.setStation(station);
    }
}
