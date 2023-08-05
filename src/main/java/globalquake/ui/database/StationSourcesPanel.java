package globalquake.ui.database;

import globalquake.ui.database.table.StationSourcesTableModel;

import javax.swing.*;
import java.awt.*;

public class StationSourcesPanel extends JPanel {
    private final DatabaseMonitorFrame databaseMonitorFrame;
    private final JTable table;

    public StationSourcesPanel(DatabaseMonitorFrame databaseMonitorFrame) {
        this.databaseMonitorFrame = databaseMonitorFrame;

        setLayout(new BorderLayout());

        JPanel actionsPanel = new JPanel();
        actionsPanel.setLayout(new GridLayout(1,1));

        actionsPanel.add(new JButton("Help"));

        add(actionsPanel, BorderLayout.NORTH);
        add(new JScrollPane(table = createTable()), BorderLayout.CENTER);
    }

    private JTable createTable() {
        StationSourcesTableModel tableModel;
        JTable table = new JTable(tableModel = new StationSourcesTableModel(databaseMonitorFrame.getManager().getStationDatabase().getStationSources()));
        table.setFont(new Font("Arial", Font.PLAIN, 14));
        table.setRowHeight(20);
        table.setGridColor(Color.black);
        table.setShowGrid(true);
        table.setAutoResizeMode(JTable.AUTO_RESIZE_ALL_COLUMNS);
        table.setAutoCreateRowSorter(true);

        for (int i = 0; i < table.getColumnCount(); i++) {
            table.getColumnModel().getColumn(i).setCellRenderer(tableModel.getColumnRenderer(i));
        }

        return table;
    }
}
