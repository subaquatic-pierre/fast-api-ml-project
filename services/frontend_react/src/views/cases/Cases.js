import React, { useState, useEffect } from 'react'
import axios from 'axios'

import { API_URL } from 'src/const'

import {
  CTable,
  CTableBody,
  CTableDataCell,
  CTableHead,
  CTableHeaderCell,
  CTableRow,
  CCardBody,
  CCard,
  CCardHeader,
} from '@coreui/react'
import { useNavigate } from 'react-router-dom'

const dummyCase = {
  id: 'string',
  type: 'string',
  userId: 'string',
  createdDate: 'string',
  status: 'PROCESSED',
  reportGenerated: true,
  topViewGenerated: false,
  reportUrl: 'string',
  reportUrlPdf: 'string',
  reportUrlPdfTopView: 'string',
  vehicleCount: 0,
  report: {},
  all_final_dict: {},
}

const Cases = () => {
  const [cases, setCases] = useState([dummyCase])

  const fetchCases = async () => {
    try {
      const url = `${API_URL}/case`
      const caseRes = await axios.get(url)
      const caseResJson = caseRes.data
      const cases = caseResJson.data
      setCases(cases)
    } catch (e) {
      console.log(e)
      setCases([])
    }
  }

  const handleDeleteCaseClick = async (caseId) => {
    const url = `${API_URL}/case/${caseId}`
    try {
      const res = await axios.delete(url)
      if (res.status === 200) {
        window.location.reload()
      }
    } catch (e) {
      console.log(e)
    }
  }

  useEffect(() => {
    fetchCases()
  }, [])

  return (
    <>
      <CCard className="mb-4">
        <CCardHeader className="d-flex align-items-center justify-content-between">
          <h5 className="mb-0">Cases</h5>
        </CCardHeader>
        <CCardBody>
          <CTable align="middle" className="mb-0 " hover responsive>
            <CTableHead color="light">
              <CTableRow>
                <CTableHeaderCell>Case ID</CTableHeaderCell>
                <CTableHeaderCell>Status</CTableHeaderCell>
                <CTableHeaderCell>User Id</CTableHeaderCell>
                <CTableHeaderCell>JSON Report</CTableHeaderCell>
                <CTableHeaderCell>PDF Report</CTableHeaderCell>
                <CTableHeaderCell className="actions"></CTableHeaderCell>
              </CTableRow>
            </CTableHead>
            <CTableBody>
              {cases.map((apiCase, index) => (
                <CTableRow v-for="item in caseItems" key={index}>
                  <CTableDataCell>
                    <div>{apiCase.id}</div>
                    <div className="small text-medium-emphasis">
                      <span>Type:</span> {apiCase.type}
                    </div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{apiCase.status}</div>
                    <div className="small text-medium-emphasis">
                      <span>Vehicle Count:</span> {apiCase.vehicleCount}
                    </div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>{apiCase.userId}</div>
                    <div className="small text-medium-emphasis">
                      <span>Created Date:</span>{' '}
                      {new Date(apiCase.createdDate * 1000).toLocaleDateString('en-UK')}
                    </div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>
                      {apiCase.reportUrl ? (
                        <a href={apiCase.reportUrl} target="_blank" rel="noreferrer">
                          Download
                        </a>
                      ) : (
                        <>Not Generated</>
                      )}
                    </div>
                  </CTableDataCell>
                  <CTableDataCell>
                    <div>
                      {apiCase.reportUrlPdf ? (
                        <a href={apiCase.reportUrlPdf} target="_blank" rel="noreferrer">
                          Download
                        </a>
                      ) : (
                        <>Not Generated</>
                      )}
                    </div>
                  </CTableDataCell>
                  <CTableDataCell className="text-end">
                    <div>
                      <a href={`/cases/${apiCase.id}`} className="btn btn-primary" type="button">
                        View
                      </a>
                      <button
                        onClick={() => handleDeleteCaseClick(apiCase.id)}
                        className="btn text-white btn-danger"
                        type="button"
                      >
                        Delete
                      </button>
                    </div>
                  </CTableDataCell>
                </CTableRow>
              ))}
            </CTableBody>
          </CTable>
        </CCardBody>
      </CCard>
    </>
  )
}

export default Cases
