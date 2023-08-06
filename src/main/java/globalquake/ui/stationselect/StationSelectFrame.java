package globalquake.ui.stationselect;

import com.morce.globalquake.database.Station;
import globalquake.ui.database.DatabaseMonitorFrame;
import globalquake.utils.monitorable.MonitorableCopyOnWriteArrayList;

import javax.swing.*;
import java.awt.*;

public class StationSelectFrame extends JDialog {

    public StationSelectFrame(DatabaseMonitorFrame owner) {
        super(owner);
        setModal(true);

        setLayout(new BorderLayout());

        JPanel panel = new StationSelectPanel(owner.getManager().getStationDatabase());

        setPreferredSize(new Dimension(1000, 800));

        add(panel, BorderLayout.CENTER);
        add(createControlPanel(), BorderLayout.EAST);

        pack();
        setLocationRelativeTo(null);
        setResizable(true);
        setTitle("Select Stations");
    }

    private JPanel createControlPanel() {
        JPanel controlPanel = new JPanel();
        controlPanel.setBackground(Color.blue);

        return controlPanel;
    }
}
